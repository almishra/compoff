#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "timing.h"
#include "cps_driver.h"
#include <omp.h>
#include <ctime>

/*
enum VARIANT {
  STATIC1, STATIC2, STATIC3, STATIC4,
  DYNAMIC1, DYNAMIC2, DYNAMIC3, DYNAMIC4,
  GUIDED1, GUIDED2, GUIDED3, GUIDED4,
  CPU, NUM_VAR
};

std::string VAR_STR[NUM_VAR] = {
  "static,1", "static,2", "static,3", "static,4",
  "dynamic,1", "dynamic,2", "dynamic,3", "dynamic,4",
  "guided,1", "guided,2", "guided,3", "guided,4", "cpu"
};
*/
enum VARIANT {
  COMBINED, COMBINED_SWAP,
  SPLIT, SPLIT_SWAP,
  COLLAPSE, COLLAPSE_SWAP,
  CPU, NUM_VAR
};

std::string VAR_STR[NUM_VAR] = {
  "combined", "combined_swap",
  "split", "split_swap",
  "collapse", "collapse_swap",
  "cpu"
};

static FILE *fp;
static long mem_to;
static long mem_from;
static int num_dev;

void output(VARIANT var, int device, double runtime)
{
#pragma omp critical
  {
    fprintf(fp, "GPU%d,%s,%ld,%ld,%d,%d,%d,%d,%.0lf\n", device, VAR_STR[var].c_str(), mem_to, mem_from, LX,LY,LZ,LT, runtime);
    fflush(fp);
  }
}

void print_in(float (*in)[LT+1][LZ+1][LY+1][(LX/2)+1][4][3][3][2])
{
  for(int t=0; t<LT+1; t++) {
    for(int z=0; z<LZ+1; z++) {
      for(int y=0; y<LY+1; y++) {
        for(int x=0; x<LX/2+1; x++) {
          for(int c1=0; c1<2; c1++) {
            for(int mu=0;mu<4;mu++) {
              for(int s=0;s<3;s++) {
                for(int c=0;c<3;c++) {
                  for(int r=0;r<2;r++) {
                    printf("%lf ", in[t][z][y][x][c1][mu][s][c][r]);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  printf("\n");
}

void print(float (*out)[LZ][LY][LX/2][4][3][2])
{
  for(int t=0; t<LT; t++) {
    for(int z=0; z<LZ; z++) {
      for(int y=0; y<LY; y++) {
        for(int x=0; x<LX; x++) {
          for(int a=0; a<4; a++) {
            for(int b=0; b<3; b++) {
              for(int c=0; c<2; c++) {
                printf("%lf ", out[t][z][y][x][a][b][c]);
              }
            }
          }
        }
      }
    }
  }
  printf("\n");
}

int compare(float (*out1)[LZ][LY][LX/2][4][3][2], float (*out2)[LZ][LY][LX/2][4][3][2])
{
  int ret = 0;
  for(int t=0; t<LT; t++) {
    for(int z=0; z<LZ; z++) {
      for(int y=0; y<LY; y++) {
        for(int x=0; x<LX; x++) {
          for(int a=0; a<4; a++) {
            for(int b=0; b<3; b++) {
              for(int c=0; c<2; c++) {
                if(out1[t][z][y][x][a][b][c] != out2[t][z][y][x][a][b][c]) {
                  printf("Mismatch %d %d %d %d %d %d %d - %lf  %lf\n", t, z, y, x, a, b, c, out1[t][z][y][x][a][b][c], out2[t][z][y][x][a][b][c]);
                  ret = -1;
                  //return -1;
                }
              }
            }
          }
        }
      }
    }
  }

  return ret;
}

void wilson_dslash(float (*chi_p)[LZ][LY][LX/2][4][3][2],
    float (*u_p_f_tst)[LT+1][LZ+1][LY+1][(LX/2)+1][4][3][3][2],
    float (*psi_p_f_tst)[LZ+2][LY+2][(LX/2)+2][4][3][2], int cb, int dev)
{
  double runtime;
  flush_llc();
  reset_timer();
  start_timer();
#pragma omp target teams distribute parallel for collapse(1) schedule(static)
  for(int t=1; t<LT+1; t++) {
    for(int z=1; z<LZ+1; z++) {
      for(int y=1; y<LY+1; y++) {
        for(int x=2; x<LX+2; x++) {
          float chi_tmp[24];
          float tmp_tst[4][3][2];
          int cbn = (cb == 0 ? 1 : 0);
          int parity = (x-2)+(y-1)+(z-1)+(t-1);
          int sdag = 1;

          parity = parity % 2;
          if(parity != cb) {
            /* x,y,z,t addressing of cb checkerboard */
            int xp = (x+1);
            int yp = (y+1);
            int zp = (z+1);
            int tp = (t+1);

            int xm = x-1;
            int ym = y-1;
            int zm = z-1;
            int tm = t-1;

            /* 1-gamma_0 */
            /*-----------*/

            int mu = 0;
            for(int c=0;c<3;c++) {
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][y][xp/2][0][c][0]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][3][c][1]);


              tmp_tst[0][c][1] = psi_p_f_tst[t][z][y][xp/2][0][c][1]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][3][c][0]);


              tmp_tst[1][c][0] = psi_p_f_tst[t][z][y][xp/2][1][c][0]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][2][c][1]);


              tmp_tst[1][c][1] = psi_p_f_tst[t][z][y][xp/2][1][c][1]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][2][c][0]);


              tmp_tst[2][c][0] = psi_p_f_tst[t][z][y][xp/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][1][c][1]);


              tmp_tst[2][c][1] = psi_p_f_tst[t][z][y][xp/2][2][c][1]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][1][c][0]);


              tmp_tst[3][c][0] = psi_p_f_tst[t][z][y][xp/2][3][c][0]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][0][c][1]);


              tmp_tst[3][c][1] = psi_p_f_tst[t][z][y][xp/2][3][c][1]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][0][c][0]);
            }
            /* multiply by U_mu */
            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                chi_tmp[c*2 + s*3*2] =  (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                chi_tmp[1 + c*2 + s*3*2]= (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1-gamma_1 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //tmp[0 + c*2 + 0*2*3]
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][yp][x/2][0][c][0]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][3][c][0]);

              //tmp[1 + c*2 + 0*2*3]
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][yp][x/2][0][c][1]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][3][c][1]);

              //tmp[0 + c*2 + 1*2*3]
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][yp][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][2][c][0]);

              //tmp[1 + c*2 + 1*2*3]
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][yp][x/2][1][c][1]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][2][c][1]);

              //tmp[0 + c*2 + 2*2*3]
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][yp][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][1][c][0]);

              //tmp[1 + c*2 + 2*2*3]
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][yp][x/2][2][c][1]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][1][c][1]);

              //tmp[0 + c*2 + 3*2*3]
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][yp][x/2][3][c][0]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][0][c][0]);

              //tmp[1 + c*2 + 3*2*3]
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][yp][x/2][3][c][1]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][0][c][1]);
            }

            /* multiply by U_mu */
            mu = 1;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1-gamma_2 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][zp][y][x/2][0][c][0]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][2][c][1]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][zp][y][x/2][0][c][1]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][2][c][0]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][zp][y][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][3][c][1]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][zp][y][x/2][1][c][1]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][3][c][0]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][zp][y][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][0][c][1]);

              // TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][zp][y][x/2][2][c][1]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][0][c][0]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][zp][y][x/2][3][c][0]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][1][c][1]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][zp][y][x/2][3][c][1]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][1][c][0]);
            }
            /* multiply by U_mu */
            mu = 2;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1]+= (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);

              }
            }

            /* 1-gamma_3 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[tp][z][y][x/2][0][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][2][c][0]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[tp][z][y][x/2][0][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][2][c][1]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[tp][z][y][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][3][c][0]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[tp][z][y][x/2][1][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][3][c][1]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[tp][z][y][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][0][c][0]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[tp][z][y][x/2][2][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][0][c][1]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[tp][z][y][x/2][3][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][1][c][0]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[tp][z][y][x/2][3][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][1][c][1]);
            }

            /* multiply by U_mu */
            mu = 3;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1+gamma_0 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][y][xm/2][0][c][0]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][3][c][1]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][y][xm/2][0][c][1]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][3][c][0]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][y][xm/2][1][c][0]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][2][c][1]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][y][xm/2][1][c][1]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][2][c][0]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][y][xm/2][2][c][0]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][1][c][1]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][y][xm/2][2][c][1]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][1][c][0]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][y][xm/2][3][c][0]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][0][c][1]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][y][xm/2][3][c][1]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][0][c][0]);
            }
            /* multiply by U_mu */
            mu = 0;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][0]
                    * tmp_tst[s][2][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][1]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][1]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][0]
                    * tmp_tst[s][2][1]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][1]
                    * tmp_tst[s][0][0]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][1]
                    * tmp_tst[s][1][0]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1+gamma_1 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][ym][x/2][0][c][0]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][3][c][0]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][ym][x/2][0][c][1]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][3][c][1]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][ym][x/2][1][c][0]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][2][c][0]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][ym][x/2][1][c][1]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][2][c][1]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][ym][x/2][2][c][0]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][1][c][0]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][ym][x/2][2][c][1]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][1][c][1]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][ym][x/2][3][c][0]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][0][c][0]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][ym][x/2][3][c][1]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][0][c][1]);
            }

            /* multiply by U_mu */
            mu = 1;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1+gamma_2 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][zm][y][x/2][0][c][0]//PSI(0,c,0)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][zm][y][x/2][0][c][1]//PSI(1,c,0)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][zm][y][x/2][1][c][0]//PSI(0,c,1)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][zm][y][x/2][1][c][1]//PSI(1,c,1)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][zm][y][x/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][0][c][1]);//PSI(1,c,0) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][zm][y][x/2][2][c][1]//PSI(1,c,2)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][zm][y][x/2][3][c][0]//PSI(0,c,3)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][1][c][1]);//PSI(1,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][zm][y][x/2][3][c][1]//PSI(1,c,3)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][1][c][0]);//PSI(0,c,1) );
            }
            /* multiply by U_mu */

            mu = 2;


            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1+gamma_3 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[tm][z][y][x/2][0][c][0]//PSI(0,c,0)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[tm][z][y][x/2][0][c][1]//PSI(1,c,0)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[tm][z][y][x/2][1][c][0]//PSI(0,c,1)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[tm][z][y][x/2][1][c][1]//PSI(1,c,1)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[tm][z][y][x/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[tm][z][y][x/2][2][c][1]//PSI(1,c,2)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][0][c][1]);//PSI(1,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[tm][z][y][x/2][3][c][0]//PSI(0,c,3)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][1][c][0]);//PSI(0,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[tm][z][y][x/2][3][c][1]//PSI(1,c,3)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][1][c][1]);//PSI(1,c,1) );
            }

            /* multiply by U_mu */
            mu = 3;
            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );

                chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0] = chi_tmp[c*2 + s*3*2];
                chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1] = chi_tmp[c*2 + s*3*2 +1];
              }
            }
          } // end if parity == cbn
        } // end t loop
      } // end z loop
    } // end y loop
  } // end x loop
  runtime = stop_timer();
  output(COMBINED, dev, runtime);
}

void wilson_dslash_collapse(float (*chi_p)[LZ][LY][LX/2][4][3][2],
    float (*u_p_f_tst)[LT+1][LZ+1][LY+1][(LX/2)+1][4][3][3][2],
    float (*psi_p_f_tst)[LZ+2][LY+2][(LX/2)+2][4][3][2], int cb, int dev)
{
  double runtime;
  flush_llc();
  reset_timer();
  start_timer();
#pragma omp target teams distribute parallel for collapse(2) schedule(static)
  for(int t=1; t<LT+1; t++) {
    for(int z=1; z<LZ+1; z++) {
      for(int y=1; y<LY+1; y++) {
        for(int x=2; x<LX+2; x++) {
          float chi_tmp[24];
          float tmp_tst[4][3][2];
          int cbn = (cb == 0 ? 1 : 0);
          int parity = (x-2)+(y-1)+(z-1)+(t-1);
          int sdag = 1;

          parity = parity % 2;
          if(parity != cb) {
            /* x,y,z,t addressing of cb checkerboard */
            int xp = (x+1);
            int yp = (y+1);
            int zp = (z+1);
            int tp = (t+1);

            int xm = x-1;
            int ym = y-1;
            int zm = z-1;
            int tm = t-1;

            /* 1-gamma_0 */
            /*-----------*/

            int mu = 0;
            for(int c=0;c<3;c++) {
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][y][xp/2][0][c][0]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][3][c][1]);


              tmp_tst[0][c][1] = psi_p_f_tst[t][z][y][xp/2][0][c][1]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][3][c][0]);


              tmp_tst[1][c][0] = psi_p_f_tst[t][z][y][xp/2][1][c][0]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][2][c][1]);


              tmp_tst[1][c][1] = psi_p_f_tst[t][z][y][xp/2][1][c][1]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][2][c][0]);


              tmp_tst[2][c][0] = psi_p_f_tst[t][z][y][xp/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][1][c][1]);


              tmp_tst[2][c][1] = psi_p_f_tst[t][z][y][xp/2][2][c][1]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][1][c][0]);


              tmp_tst[3][c][0] = psi_p_f_tst[t][z][y][xp/2][3][c][0]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][0][c][1]);


              tmp_tst[3][c][1] = psi_p_f_tst[t][z][y][xp/2][3][c][1]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][0][c][0]);
            }
            /* multiply by U_mu */
            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                chi_tmp[c*2 + s*3*2] =  (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                chi_tmp[1 + c*2 + s*3*2]= (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1-gamma_1 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //tmp[0 + c*2 + 0*2*3]
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][yp][x/2][0][c][0]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][3][c][0]);

              //tmp[1 + c*2 + 0*2*3]
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][yp][x/2][0][c][1]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][3][c][1]);

              //tmp[0 + c*2 + 1*2*3]
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][yp][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][2][c][0]);

              //tmp[1 + c*2 + 1*2*3]
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][yp][x/2][1][c][1]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][2][c][1]);

              //tmp[0 + c*2 + 2*2*3]
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][yp][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][1][c][0]);

              //tmp[1 + c*2 + 2*2*3]
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][yp][x/2][2][c][1]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][1][c][1]);

              //tmp[0 + c*2 + 3*2*3]
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][yp][x/2][3][c][0]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][0][c][0]);

              //tmp[1 + c*2 + 3*2*3]
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][yp][x/2][3][c][1]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][0][c][1]);
            }

            /* multiply by U_mu */
            mu = 1;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1-gamma_2 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][zp][y][x/2][0][c][0]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][2][c][1]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][zp][y][x/2][0][c][1]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][2][c][0]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][zp][y][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][3][c][1]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][zp][y][x/2][1][c][1]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][3][c][0]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][zp][y][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][0][c][1]);

              // TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][zp][y][x/2][2][c][1]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][0][c][0]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][zp][y][x/2][3][c][0]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][1][c][1]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][zp][y][x/2][3][c][1]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][1][c][0]);
            }
            /* multiply by U_mu */
            mu = 2;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1]+= (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);

              }
            }

            /* 1-gamma_3 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[tp][z][y][x/2][0][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][2][c][0]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[tp][z][y][x/2][0][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][2][c][1]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[tp][z][y][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][3][c][0]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[tp][z][y][x/2][1][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][3][c][1]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[tp][z][y][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][0][c][0]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[tp][z][y][x/2][2][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][0][c][1]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[tp][z][y][x/2][3][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][1][c][0]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[tp][z][y][x/2][3][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][1][c][1]);
            }

            /* multiply by U_mu */
            mu = 3;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1+gamma_0 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][y][xm/2][0][c][0]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][3][c][1]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][y][xm/2][0][c][1]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][3][c][0]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][y][xm/2][1][c][0]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][2][c][1]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][y][xm/2][1][c][1]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][2][c][0]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][y][xm/2][2][c][0]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][1][c][1]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][y][xm/2][2][c][1]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][1][c][0]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][y][xm/2][3][c][0]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][0][c][1]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][y][xm/2][3][c][1]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][0][c][0]);
            }
            /* multiply by U_mu */
            mu = 0;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][0]
                    * tmp_tst[s][2][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][1]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][1]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][0]
                    * tmp_tst[s][2][1]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][1]
                    * tmp_tst[s][0][0]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][1]
                    * tmp_tst[s][1][0]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1+gamma_1 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][ym][x/2][0][c][0]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][3][c][0]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][ym][x/2][0][c][1]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][3][c][1]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][ym][x/2][1][c][0]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][2][c][0]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][ym][x/2][1][c][1]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][2][c][1]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][ym][x/2][2][c][0]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][1][c][0]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][ym][x/2][2][c][1]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][1][c][1]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][ym][x/2][3][c][0]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][0][c][0]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][ym][x/2][3][c][1]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][0][c][1]);
            }

            /* multiply by U_mu */
            mu = 1;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1+gamma_2 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][zm][y][x/2][0][c][0]//PSI(0,c,0)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][zm][y][x/2][0][c][1]//PSI(1,c,0)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][zm][y][x/2][1][c][0]//PSI(0,c,1)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][zm][y][x/2][1][c][1]//PSI(1,c,1)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][zm][y][x/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][0][c][1]);//PSI(1,c,0) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][zm][y][x/2][2][c][1]//PSI(1,c,2)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][zm][y][x/2][3][c][0]//PSI(0,c,3)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][1][c][1]);//PSI(1,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][zm][y][x/2][3][c][1]//PSI(1,c,3)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][1][c][0]);//PSI(0,c,1) );
            }
            /* multiply by U_mu */

            mu = 2;


            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1+gamma_3 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[tm][z][y][x/2][0][c][0]//PSI(0,c,0)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[tm][z][y][x/2][0][c][1]//PSI(1,c,0)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[tm][z][y][x/2][1][c][0]//PSI(0,c,1)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[tm][z][y][x/2][1][c][1]//PSI(1,c,1)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[tm][z][y][x/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[tm][z][y][x/2][2][c][1]//PSI(1,c,2)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][0][c][1]);//PSI(1,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[tm][z][y][x/2][3][c][0]//PSI(0,c,3)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][1][c][0]);//PSI(0,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[tm][z][y][x/2][3][c][1]//PSI(1,c,3)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][1][c][1]);//PSI(1,c,1) );
            }

            /* multiply by U_mu */
            mu = 3;
            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );

                chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0] = chi_tmp[c*2 + s*3*2];
                chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1] = chi_tmp[c*2 + s*3*2 +1];
              }
            }
          } // end if parity == cbn
        } // end t loop
      } // end z loop
    } // end y loop
  } // end x loop
  runtime = stop_timer();
  output(COLLAPSE, dev, runtime);
}

void wilson_dslash_split(float (*chi_p)[LZ][LY][LX/2][4][3][2],
    float (*u_p_f_tst)[LT+1][LZ+1][LY+1][(LX/2)+1][4][3][3][2],
    float (*psi_p_f_tst)[LZ+2][LY+2][(LX/2)+2][4][3][2], int cb, int dev)
{
  double runtime;
  flush_llc();
  reset_timer();
  start_timer();
#pragma omp target teams distribute
  for(int t=1; t<LT+1; t++) {
#pragma omp parallel for
    for(int z=1; z<LZ+1; z++) {
      for(int y=1; y<LY+1; y++) {
        for(int x=2; x<LX+2; x++) {
          float chi_tmp[24];
          float tmp_tst[4][3][2];
          int cbn = (cb == 0 ? 1 : 0);
          int parity = (x-2)+(y-1)+(z-1)+(t-1);
          int sdag = 1;

          parity = parity % 2;
          if(parity != cb) {
            /* x,y,z,t addressing of cb checkerboard */
            int xp = (x+1);
            int yp = (y+1);
            int zp = (z+1);
            int tp = (t+1);

            int xm = x-1;
            int ym = y-1;
            int zm = z-1;
            int tm = t-1;

            /* 1-gamma_0 */
            /*-----------*/

            int mu = 0;
            for(int c=0;c<3;c++) {
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][y][xp/2][0][c][0]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][3][c][1]);


              tmp_tst[0][c][1] = psi_p_f_tst[t][z][y][xp/2][0][c][1]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][3][c][0]);


              tmp_tst[1][c][0] = psi_p_f_tst[t][z][y][xp/2][1][c][0]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][2][c][1]);


              tmp_tst[1][c][1] = psi_p_f_tst[t][z][y][xp/2][1][c][1]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][2][c][0]);


              tmp_tst[2][c][0] = psi_p_f_tst[t][z][y][xp/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][1][c][1]);


              tmp_tst[2][c][1] = psi_p_f_tst[t][z][y][xp/2][2][c][1]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][1][c][0]);


              tmp_tst[3][c][0] = psi_p_f_tst[t][z][y][xp/2][3][c][0]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][0][c][1]);


              tmp_tst[3][c][1] = psi_p_f_tst[t][z][y][xp/2][3][c][1]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][0][c][0]);
            }
            /* multiply by U_mu */
            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                chi_tmp[c*2 + s*3*2] =  (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                chi_tmp[1 + c*2 + s*3*2]= (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1-gamma_1 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //tmp[0 + c*2 + 0*2*3]
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][yp][x/2][0][c][0]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][3][c][0]);

              //tmp[1 + c*2 + 0*2*3]
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][yp][x/2][0][c][1]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][3][c][1]);

              //tmp[0 + c*2 + 1*2*3]
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][yp][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][2][c][0]);

              //tmp[1 + c*2 + 1*2*3]
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][yp][x/2][1][c][1]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][2][c][1]);

              //tmp[0 + c*2 + 2*2*3]
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][yp][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][1][c][0]);

              //tmp[1 + c*2 + 2*2*3]
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][yp][x/2][2][c][1]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][1][c][1]);

              //tmp[0 + c*2 + 3*2*3]
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][yp][x/2][3][c][0]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][0][c][0]);

              //tmp[1 + c*2 + 3*2*3]
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][yp][x/2][3][c][1]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][0][c][1]);
            }

            /* multiply by U_mu */
            mu = 1;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1-gamma_2 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][zp][y][x/2][0][c][0]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][2][c][1]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][zp][y][x/2][0][c][1]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][2][c][0]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][zp][y][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][3][c][1]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][zp][y][x/2][1][c][1]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][3][c][0]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][zp][y][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][0][c][1]);

              // TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][zp][y][x/2][2][c][1]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][0][c][0]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][zp][y][x/2][3][c][0]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][1][c][1]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][zp][y][x/2][3][c][1]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][1][c][0]);
            }
            /* multiply by U_mu */
            mu = 2;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1]+= (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);

              }
            }

            /* 1-gamma_3 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[tp][z][y][x/2][0][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][2][c][0]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[tp][z][y][x/2][0][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][2][c][1]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[tp][z][y][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][3][c][0]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[tp][z][y][x/2][1][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][3][c][1]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[tp][z][y][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][0][c][0]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[tp][z][y][x/2][2][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][0][c][1]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[tp][z][y][x/2][3][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][1][c][0]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[tp][z][y][x/2][3][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][1][c][1]);
            }

            /* multiply by U_mu */
            mu = 3;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1+gamma_0 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][y][xm/2][0][c][0]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][3][c][1]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][y][xm/2][0][c][1]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][3][c][0]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][y][xm/2][1][c][0]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][2][c][1]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][y][xm/2][1][c][1]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][2][c][0]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][y][xm/2][2][c][0]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][1][c][1]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][y][xm/2][2][c][1]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][1][c][0]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][y][xm/2][3][c][0]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][0][c][1]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][y][xm/2][3][c][1]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][0][c][0]);
            }
            /* multiply by U_mu */
            mu = 0;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][0]
                    * tmp_tst[s][2][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][1]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][1]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][0]
                    * tmp_tst[s][2][1]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][1]
                    * tmp_tst[s][0][0]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][1]
                    * tmp_tst[s][1][0]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1+gamma_1 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][ym][x/2][0][c][0]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][3][c][0]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][ym][x/2][0][c][1]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][3][c][1]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][ym][x/2][1][c][0]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][2][c][0]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][ym][x/2][1][c][1]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][2][c][1]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][ym][x/2][2][c][0]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][1][c][0]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][ym][x/2][2][c][1]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][1][c][1]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][ym][x/2][3][c][0]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][0][c][0]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][ym][x/2][3][c][1]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][0][c][1]);
            }

            /* multiply by U_mu */
            mu = 1;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1+gamma_2 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][zm][y][x/2][0][c][0]//PSI(0,c,0)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][zm][y][x/2][0][c][1]//PSI(1,c,0)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][zm][y][x/2][1][c][0]//PSI(0,c,1)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][zm][y][x/2][1][c][1]//PSI(1,c,1)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][zm][y][x/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][0][c][1]);//PSI(1,c,0) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][zm][y][x/2][2][c][1]//PSI(1,c,2)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][zm][y][x/2][3][c][0]//PSI(0,c,3)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][1][c][1]);//PSI(1,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][zm][y][x/2][3][c][1]//PSI(1,c,3)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][1][c][0]);//PSI(0,c,1) );
            }
            /* multiply by U_mu */

            mu = 2;


            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1+gamma_3 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[tm][z][y][x/2][0][c][0]//PSI(0,c,0)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[tm][z][y][x/2][0][c][1]//PSI(1,c,0)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[tm][z][y][x/2][1][c][0]//PSI(0,c,1)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[tm][z][y][x/2][1][c][1]//PSI(1,c,1)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[tm][z][y][x/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[tm][z][y][x/2][2][c][1]//PSI(1,c,2)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][0][c][1]);//PSI(1,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[tm][z][y][x/2][3][c][0]//PSI(0,c,3)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][1][c][0]);//PSI(0,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[tm][z][y][x/2][3][c][1]//PSI(1,c,3)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][1][c][1]);//PSI(1,c,1) );
            }

            /* multiply by U_mu */
            mu = 3;
            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );

                chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0] = chi_tmp[c*2 + s*3*2];
                chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1] = chi_tmp[c*2 + s*3*2 +1];
              }
            }
          } // end if parity == cbn
        } // end t loop
      } // end z loop
    } // end y loop
  } // end x loop
  runtime = stop_timer();
  output(SPLIT, dev, runtime);
}

void wilson_dslash_swap(float (*chi_p)[LZ][LY][LX/2][4][3][2],
    float (*u_p_f_tst)[LT+1][LZ+1][LY+1][(LX/2)+1][4][3][3][2],
    float (*psi_p_f_tst)[LZ+2][LY+2][(LX/2)+2][4][3][2], int cb, int dev)
{
  double runtime;
  flush_llc();
  reset_timer();
  start_timer();
#pragma omp target teams distribute parallel for 
  for(int z=1; z<LZ+1; z++) {
    for(int t=1; t<LT+1; t++) {
      for(int y=1; y<LY+1; y++) {
        for(int x=2; x<LX+2; x++) {
          float chi_tmp[24];
          float tmp_tst[4][3][2];
          int cbn = (cb == 0 ? 1 : 0);
          int parity = (x-2)+(y-1)+(z-1)+(t-1);
          int sdag = 1;

          parity = parity % 2;
          if(parity != cb) {
            /* x,y,z,t addressing of cb checkerboard */
            int xp = (x+1);
            int yp = (y+1);
            int zp = (z+1);
            int tp = (t+1);

            int xm = x-1;
            int ym = y-1;
            int zm = z-1;
            int tm = t-1;

            /* 1-gamma_0 */
            /*-----------*/

            int mu = 0;
            for(int c=0;c<3;c++) {
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][y][xp/2][0][c][0]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][3][c][1]);


              tmp_tst[0][c][1] = psi_p_f_tst[t][z][y][xp/2][0][c][1]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][3][c][0]);


              tmp_tst[1][c][0] = psi_p_f_tst[t][z][y][xp/2][1][c][0]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][2][c][1]);


              tmp_tst[1][c][1] = psi_p_f_tst[t][z][y][xp/2][1][c][1]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][2][c][0]);


              tmp_tst[2][c][0] = psi_p_f_tst[t][z][y][xp/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][1][c][1]);


              tmp_tst[2][c][1] = psi_p_f_tst[t][z][y][xp/2][2][c][1]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][1][c][0]);


              tmp_tst[3][c][0] = psi_p_f_tst[t][z][y][xp/2][3][c][0]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][0][c][1]);


              tmp_tst[3][c][1] = psi_p_f_tst[t][z][y][xp/2][3][c][1]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][0][c][0]);
            }
            /* multiply by U_mu */
            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                chi_tmp[c*2 + s*3*2] =  (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                chi_tmp[1 + c*2 + s*3*2]= (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1-gamma_1 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //tmp[0 + c*2 + 0*2*3]
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][yp][x/2][0][c][0]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][3][c][0]);

              //tmp[1 + c*2 + 0*2*3]
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][yp][x/2][0][c][1]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][3][c][1]);

              //tmp[0 + c*2 + 1*2*3]
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][yp][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][2][c][0]);

              //tmp[1 + c*2 + 1*2*3]
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][yp][x/2][1][c][1]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][2][c][1]);

              //tmp[0 + c*2 + 2*2*3]
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][yp][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][1][c][0]);

              //tmp[1 + c*2 + 2*2*3]
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][yp][x/2][2][c][1]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][1][c][1]);

              //tmp[0 + c*2 + 3*2*3]
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][yp][x/2][3][c][0]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][0][c][0]);

              //tmp[1 + c*2 + 3*2*3]
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][yp][x/2][3][c][1]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][0][c][1]);
            }

            /* multiply by U_mu */
            mu = 1;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1-gamma_2 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][zp][y][x/2][0][c][0]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][2][c][1]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][zp][y][x/2][0][c][1]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][2][c][0]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][zp][y][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][3][c][1]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][zp][y][x/2][1][c][1]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][3][c][0]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][zp][y][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][0][c][1]);

              // TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][zp][y][x/2][2][c][1]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][0][c][0]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][zp][y][x/2][3][c][0]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][1][c][1]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][zp][y][x/2][3][c][1]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][1][c][0]);
            }
            /* multiply by U_mu */
            mu = 2;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1]+= (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);

              }
            }

            /* 1-gamma_3 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[tp][z][y][x/2][0][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][2][c][0]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[tp][z][y][x/2][0][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][2][c][1]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[tp][z][y][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][3][c][0]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[tp][z][y][x/2][1][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][3][c][1]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[tp][z][y][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][0][c][0]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[tp][z][y][x/2][2][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][0][c][1]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[tp][z][y][x/2][3][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][1][c][0]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[tp][z][y][x/2][3][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][1][c][1]);
            }

            /* multiply by U_mu */
            mu = 3;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1+gamma_0 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][y][xm/2][0][c][0]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][3][c][1]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][y][xm/2][0][c][1]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][3][c][0]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][y][xm/2][1][c][0]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][2][c][1]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][y][xm/2][1][c][1]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][2][c][0]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][y][xm/2][2][c][0]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][1][c][1]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][y][xm/2][2][c][1]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][1][c][0]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][y][xm/2][3][c][0]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][0][c][1]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][y][xm/2][3][c][1]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][0][c][0]);
            }
            /* multiply by U_mu */
            mu = 0;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][0]
                    * tmp_tst[s][2][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][1]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][1]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][0]
                    * tmp_tst[s][2][1]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][1]
                    * tmp_tst[s][0][0]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][1]
                    * tmp_tst[s][1][0]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1+gamma_1 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][ym][x/2][0][c][0]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][3][c][0]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][ym][x/2][0][c][1]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][3][c][1]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][ym][x/2][1][c][0]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][2][c][0]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][ym][x/2][1][c][1]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][2][c][1]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][ym][x/2][2][c][0]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][1][c][0]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][ym][x/2][2][c][1]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][1][c][1]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][ym][x/2][3][c][0]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][0][c][0]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][ym][x/2][3][c][1]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][0][c][1]);
            }

            /* multiply by U_mu */
            mu = 1;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1+gamma_2 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][zm][y][x/2][0][c][0]//PSI(0,c,0)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][zm][y][x/2][0][c][1]//PSI(1,c,0)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][zm][y][x/2][1][c][0]//PSI(0,c,1)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][zm][y][x/2][1][c][1]//PSI(1,c,1)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][zm][y][x/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][0][c][1]);//PSI(1,c,0) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][zm][y][x/2][2][c][1]//PSI(1,c,2)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][zm][y][x/2][3][c][0]//PSI(0,c,3)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][1][c][1]);//PSI(1,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][zm][y][x/2][3][c][1]//PSI(1,c,3)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][1][c][0]);//PSI(0,c,1) );
            }
            /* multiply by U_mu */

            mu = 2;


            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1+gamma_3 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[tm][z][y][x/2][0][c][0]//PSI(0,c,0)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[tm][z][y][x/2][0][c][1]//PSI(1,c,0)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[tm][z][y][x/2][1][c][0]//PSI(0,c,1)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[tm][z][y][x/2][1][c][1]//PSI(1,c,1)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[tm][z][y][x/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[tm][z][y][x/2][2][c][1]//PSI(1,c,2)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][0][c][1]);//PSI(1,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[tm][z][y][x/2][3][c][0]//PSI(0,c,3)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][1][c][0]);//PSI(0,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[tm][z][y][x/2][3][c][1]//PSI(1,c,3)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][1][c][1]);//PSI(1,c,1) );
            }

            /* multiply by U_mu */
            mu = 3;
            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );

                chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0] = chi_tmp[c*2 + s*3*2];
                chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1] = chi_tmp[c*2 + s*3*2 +1];
              }
            }
          } // end if parity == cbn
        } // end t loop
      } // end z loop
    } // end y loop
  } // end x loop
  runtime = stop_timer();
  output(COMBINED_SWAP, dev, runtime);
}

void wilson_dslash_collapse_swap(float (*chi_p)[LZ][LY][LX/2][4][3][2],
    float (*u_p_f_tst)[LT+1][LZ+1][LY+1][(LX/2)+1][4][3][3][2],
    float (*psi_p_f_tst)[LZ+2][LY+2][(LX/2)+2][4][3][2], int cb, int dev)
{
  double runtime;
  flush_llc();
  reset_timer();
  start_timer();
#pragma omp target teams distribute parallel for collapse(2)
  for(int z=1; z<LZ+1; z++) {
    for(int t=1; t<LT+1; t++) {
      for(int y=1; y<LY+1; y++) {
        for(int x=2; x<LX+2; x++) {
          float chi_tmp[24];
          float tmp_tst[4][3][2];
          int cbn = (cb == 0 ? 1 : 0);
          int parity = (x-2)+(y-1)+(z-1)+(t-1);
          int sdag = 1;

          parity = parity % 2;
          if(parity != cb) {
            /* x,y,z,t addressing of cb checkerboard */
            int xp = (x+1);
            int yp = (y+1);
            int zp = (z+1);
            int tp = (t+1);

            int xm = x-1;
            int ym = y-1;
            int zm = z-1;
            int tm = t-1;

            /* 1-gamma_0 */
            /*-----------*/

            int mu = 0;
            for(int c=0;c<3;c++) {
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][y][xp/2][0][c][0]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][3][c][1]);


              tmp_tst[0][c][1] = psi_p_f_tst[t][z][y][xp/2][0][c][1]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][3][c][0]);


              tmp_tst[1][c][0] = psi_p_f_tst[t][z][y][xp/2][1][c][0]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][2][c][1]);


              tmp_tst[1][c][1] = psi_p_f_tst[t][z][y][xp/2][1][c][1]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][2][c][0]);


              tmp_tst[2][c][0] = psi_p_f_tst[t][z][y][xp/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][1][c][1]);


              tmp_tst[2][c][1] = psi_p_f_tst[t][z][y][xp/2][2][c][1]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][1][c][0]);


              tmp_tst[3][c][0] = psi_p_f_tst[t][z][y][xp/2][3][c][0]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][0][c][1]);


              tmp_tst[3][c][1] = psi_p_f_tst[t][z][y][xp/2][3][c][1]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][0][c][0]);
            }
            /* multiply by U_mu */
            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                chi_tmp[c*2 + s*3*2] =  (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                chi_tmp[1 + c*2 + s*3*2]= (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1-gamma_1 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //tmp[0 + c*2 + 0*2*3]
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][yp][x/2][0][c][0]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][3][c][0]);

              //tmp[1 + c*2 + 0*2*3]
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][yp][x/2][0][c][1]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][3][c][1]);

              //tmp[0 + c*2 + 1*2*3]
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][yp][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][2][c][0]);

              //tmp[1 + c*2 + 1*2*3]
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][yp][x/2][1][c][1]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][2][c][1]);

              //tmp[0 + c*2 + 2*2*3]
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][yp][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][1][c][0]);

              //tmp[1 + c*2 + 2*2*3]
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][yp][x/2][2][c][1]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][1][c][1]);

              //tmp[0 + c*2 + 3*2*3]
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][yp][x/2][3][c][0]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][0][c][0]);

              //tmp[1 + c*2 + 3*2*3]
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][yp][x/2][3][c][1]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][0][c][1]);
            }

            /* multiply by U_mu */
            mu = 1;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1-gamma_2 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][zp][y][x/2][0][c][0]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][2][c][1]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][zp][y][x/2][0][c][1]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][2][c][0]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][zp][y][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][3][c][1]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][zp][y][x/2][1][c][1]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][3][c][0]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][zp][y][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][0][c][1]);

              // TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][zp][y][x/2][2][c][1]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][0][c][0]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][zp][y][x/2][3][c][0]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][1][c][1]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][zp][y][x/2][3][c][1]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][1][c][0]);
            }
            /* multiply by U_mu */
            mu = 2;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1]+= (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);

              }
            }

            /* 1-gamma_3 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[tp][z][y][x/2][0][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][2][c][0]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[tp][z][y][x/2][0][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][2][c][1]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[tp][z][y][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][3][c][0]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[tp][z][y][x/2][1][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][3][c][1]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[tp][z][y][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][0][c][0]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[tp][z][y][x/2][2][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][0][c][1]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[tp][z][y][x/2][3][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][1][c][0]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[tp][z][y][x/2][3][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][1][c][1]);
            }

            /* multiply by U_mu */
            mu = 3;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1+gamma_0 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][y][xm/2][0][c][0]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][3][c][1]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][y][xm/2][0][c][1]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][3][c][0]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][y][xm/2][1][c][0]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][2][c][1]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][y][xm/2][1][c][1]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][2][c][0]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][y][xm/2][2][c][0]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][1][c][1]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][y][xm/2][2][c][1]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][1][c][0]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][y][xm/2][3][c][0]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][0][c][1]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][y][xm/2][3][c][1]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][0][c][0]);
            }
            /* multiply by U_mu */
            mu = 0;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][0]
                    * tmp_tst[s][2][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][1]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][1]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][0]
                    * tmp_tst[s][2][1]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][1]
                    * tmp_tst[s][0][0]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][1]
                    * tmp_tst[s][1][0]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1+gamma_1 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][ym][x/2][0][c][0]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][3][c][0]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][ym][x/2][0][c][1]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][3][c][1]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][ym][x/2][1][c][0]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][2][c][0]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][ym][x/2][1][c][1]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][2][c][1]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][ym][x/2][2][c][0]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][1][c][0]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][ym][x/2][2][c][1]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][1][c][1]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][ym][x/2][3][c][0]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][0][c][0]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][ym][x/2][3][c][1]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][0][c][1]);
            }

            /* multiply by U_mu */
            mu = 1;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1+gamma_2 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][zm][y][x/2][0][c][0]//PSI(0,c,0)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][zm][y][x/2][0][c][1]//PSI(1,c,0)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][zm][y][x/2][1][c][0]//PSI(0,c,1)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][zm][y][x/2][1][c][1]//PSI(1,c,1)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][zm][y][x/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][0][c][1]);//PSI(1,c,0) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][zm][y][x/2][2][c][1]//PSI(1,c,2)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][zm][y][x/2][3][c][0]//PSI(0,c,3)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][1][c][1]);//PSI(1,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][zm][y][x/2][3][c][1]//PSI(1,c,3)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][1][c][0]);//PSI(0,c,1) );
            }
            /* multiply by U_mu */

            mu = 2;


            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1+gamma_3 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[tm][z][y][x/2][0][c][0]//PSI(0,c,0)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[tm][z][y][x/2][0][c][1]//PSI(1,c,0)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[tm][z][y][x/2][1][c][0]//PSI(0,c,1)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[tm][z][y][x/2][1][c][1]//PSI(1,c,1)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[tm][z][y][x/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[tm][z][y][x/2][2][c][1]//PSI(1,c,2)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][0][c][1]);//PSI(1,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[tm][z][y][x/2][3][c][0]//PSI(0,c,3)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][1][c][0]);//PSI(0,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[tm][z][y][x/2][3][c][1]//PSI(1,c,3)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][1][c][1]);//PSI(1,c,1) );
            }

            /* multiply by U_mu */
            mu = 3;
            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );

                chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0] = chi_tmp[c*2 + s*3*2];
                chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1] = chi_tmp[c*2 + s*3*2 +1];
              }
            }
          } // end if parity == cbn
        } // end t loop
      } // end z loop
    } // end y loop
  } // end x loop
  runtime = stop_timer();
  output(COLLAPSE_SWAP, dev, runtime);
}

void wilson_dslash_split_swap(float (*chi_p)[LZ][LY][LX/2][4][3][2],
    float (*u_p_f_tst)[LT+1][LZ+1][LY+1][(LX/2)+1][4][3][3][2],
    float (*psi_p_f_tst)[LZ+2][LY+2][(LX/2)+2][4][3][2], int cb, int dev)
{
  double runtime;
  flush_llc();
  reset_timer();
  start_timer();
#pragma omp target teams distribute
  for(int z=1; z<LZ+1; z++) {
#pragma omp parallel for
    for(int t=1; t<LT+1; t++) {
      for(int y=1; y<LY+1; y++) {
        for(int x=2; x<LX+2; x++) {
          float chi_tmp[24];
          float tmp_tst[4][3][2];
          int cbn = (cb == 0 ? 1 : 0);
          int parity = (x-2)+(y-1)+(z-1)+(t-1);
          int sdag = 1;

          parity = parity % 2;
          if(parity != cb) {
            /* x,y,z,t addressing of cb checkerboard */
            int xp = (x+1);
            int yp = (y+1);
            int zp = (z+1);
            int tp = (t+1);

            int xm = x-1;
            int ym = y-1;
            int zm = z-1;
            int tm = t-1;

            /* 1-gamma_0 */
            /*-----------*/

            int mu = 0;
            for(int c=0;c<3;c++) {
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][y][xp/2][0][c][0]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][3][c][1]);


              tmp_tst[0][c][1] = psi_p_f_tst[t][z][y][xp/2][0][c][1]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][3][c][0]);


              tmp_tst[1][c][0] = psi_p_f_tst[t][z][y][xp/2][1][c][0]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][2][c][1]);


              tmp_tst[1][c][1] = psi_p_f_tst[t][z][y][xp/2][1][c][1]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][2][c][0]);


              tmp_tst[2][c][0] = psi_p_f_tst[t][z][y][xp/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][1][c][1]);


              tmp_tst[2][c][1] = psi_p_f_tst[t][z][y][xp/2][2][c][1]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][1][c][0]);


              tmp_tst[3][c][0] = psi_p_f_tst[t][z][y][xp/2][3][c][0]
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][0][c][1]);


              tmp_tst[3][c][1] = psi_p_f_tst[t][z][y][xp/2][3][c][1]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][0][c][0]);
            }
            /* multiply by U_mu */
            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                chi_tmp[c*2 + s*3*2] =  (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                chi_tmp[1 + c*2 + s*3*2]= (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1-gamma_1 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //tmp[0 + c*2 + 0*2*3]
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][yp][x/2][0][c][0]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][3][c][0]);

              //tmp[1 + c*2 + 0*2*3]
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][yp][x/2][0][c][1]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][3][c][1]);

              //tmp[0 + c*2 + 1*2*3]
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][yp][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][2][c][0]);

              //tmp[1 + c*2 + 1*2*3]
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][yp][x/2][1][c][1]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][2][c][1]);

              //tmp[0 + c*2 + 2*2*3]
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][yp][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][1][c][0]);

              //tmp[1 + c*2 + 2*2*3]
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][yp][x/2][2][c][1]
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][1][c][1]);

              //tmp[0 + c*2 + 3*2*3]
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][yp][x/2][3][c][0]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][0][c][0]);

              //tmp[1 + c*2 + 3*2*3]
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][yp][x/2][3][c][1]
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][0][c][1]);
            }

            /* multiply by U_mu */
            mu = 1;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1-gamma_2 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][zp][y][x/2][0][c][0]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][2][c][1]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][zp][y][x/2][0][c][1]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][2][c][0]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][zp][y][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][3][c][1]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][zp][y][x/2][1][c][1]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][3][c][0]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][zp][y][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][0][c][1]);

              // TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][zp][y][x/2][2][c][1]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][0][c][0]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][zp][y][x/2][3][c][0]
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][1][c][1]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][zp][y][x/2][3][c][1]
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][1][c][0]);
            }
            /* multiply by U_mu */
            mu = 2;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1]+= (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);

              }
            }

            /* 1-gamma_3 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[tp][z][y][x/2][0][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][2][c][0]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[tp][z][y][x/2][0][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][2][c][1]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[tp][z][y][x/2][1][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][3][c][0]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[tp][z][y][x/2][1][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][3][c][1]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[tp][z][y][x/2][2][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][0][c][0]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[tp][z][y][x/2][2][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][0][c][1]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[tp][z][y][x/2][3][c][0]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][1][c][0]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[tp][z][y][x/2][3][c][1]
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][1][c][1]);
            }

            /* multiply by U_mu */
            mu = 3;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1+gamma_0 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][y][xm/2][0][c][0]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][3][c][1]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][y][xm/2][0][c][1]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][3][c][0]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][y][xm/2][1][c][0]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][2][c][1]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][y][xm/2][1][c][1]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][2][c][0]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][y][xm/2][2][c][0]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][1][c][1]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][y][xm/2][2][c][1]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][1][c][0]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][y][xm/2][3][c][0]
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][0][c][1]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][y][xm/2][3][c][1]
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][0][c][0]);
            }
            /* multiply by U_mu */
            mu = 0;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][0]
                    * tmp_tst[s][0][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][0]
                    * tmp_tst[s][1][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][0]
                    * tmp_tst[s][2][0]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][1]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][1]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][0]
                    * tmp_tst[s][0][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][0]
                    * tmp_tst[s][1][1]

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][0]
                    * tmp_tst[s][2][1]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][1]
                    * tmp_tst[s][0][0]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][1]
                    * tmp_tst[s][1][0]

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);
              }
            }

            /* 1+gamma_1 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][ym][x/2][0][c][0]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][3][c][0]);

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][ym][x/2][0][c][1]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][3][c][1]);

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][ym][x/2][1][c][0]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][2][c][0]);

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][ym][x/2][1][c][1]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][2][c][1]);

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][ym][x/2][2][c][0]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][1][c][0]);

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][ym][x/2][2][c][1]
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][1][c][1]);

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][ym][x/2][3][c][0]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][0][c][0]);

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][ym][x/2][3][c][1]
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][0][c][1]);
            }

            /* multiply by U_mu */
            mu = 1;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1+gamma_2 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][zm][y][x/2][0][c][0]//PSI(0,c,0)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][zm][y][x/2][0][c][1]//PSI(1,c,0)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][zm][y][x/2][1][c][0]//PSI(0,c,1)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][zm][y][x/2][1][c][1]//PSI(1,c,1)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][zm][y][x/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][0][c][1]);//PSI(1,c,0) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][zm][y][x/2][2][c][1]//PSI(1,c,2)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][zm][y][x/2][3][c][0]//PSI(0,c,3)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][1][c][1]);//PSI(1,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][zm][y][x/2][3][c][1]//PSI(1,c,3)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][1][c][0]);//PSI(0,c,1) );
            }
            /* multiply by U_mu */

            mu = 2;


            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1+gamma_3 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[tm][z][y][x/2][0][c][0]//PSI(0,c,0)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[tm][z][y][x/2][0][c][1]//PSI(1,c,0)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[tm][z][y][x/2][1][c][0]//PSI(0,c,1)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[tm][z][y][x/2][1][c][1]//PSI(1,c,1)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[tm][z][y][x/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[tm][z][y][x/2][2][c][1]//PSI(1,c,2)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][0][c][1]);//PSI(1,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[tm][z][y][x/2][3][c][0]//PSI(0,c,3)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][1][c][0]);//PSI(0,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[tm][z][y][x/2][3][c][1]//PSI(1,c,3)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][1][c][1]);//PSI(1,c,1) );
            }

            /* multiply by U_mu */
            mu = 3;
            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );

                chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0] = chi_tmp[c*2 + s*3*2];
                chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1] = chi_tmp[c*2 + s*3*2 +1];
              }
            }
          } // end if parity == cbn
        } // end t loop
      } // end z loop
    } // end y loop
  } // end x loop
  runtime = stop_timer();
  output(SPLIT_SWAP, dev, runtime);
}

void wilson_dslash_omp(float (*chi_p)[LZ][LY][LX/2][4][3][2],
    float (*u_p_f_tst)[LT+1][LZ+1][LY+1][(LX/2)+1][4][3][3][2],
    float (*psi_p_f_tst)[LZ+2][LY+2][(LX/2)+2][4][3][2], int cb)
{
  //print_in(u_p_f_tst);
//#pragma omp parallel for
  for(int t=1; t<LT+1; t++) {
    for(int z=1; z<LZ+1; z++) {
      for(int y=1; y<LY+1; y++) {
        for(int x=2; x<LX+2; x++) {
          float chi_tmp[24];
          float tmp_tst[4][3][2];
          int cbn = (cb == 0 ? 1 : 0);
          int parity = (x-2)+(y-1)+(z-1)+(t-1);
          int sdag = 1;

          parity = parity % 2;
          if(parity != cb) {
            /* x,y,z,t addressing of cb checkerboard */
            int xp = (x+1);// - (x / (lx-1))*lx;
            int yp = (y+1);// - (y / (ly-1))*ly;
            int zp = (z+1);// - (z / (lz-1))*lz;
            int tp = (t+1);// - (t / (lt-1))*lt;

            int xm = x-1;// + ( (lx-x)/lx ) * lx;
            int ym = y-1;// + ( (ly-y)/ly ) * ly;
            int zm = z-1;// + ( (lz-z)/lz ) * lz;
            int tm = t-1;// + ( (lt-t)/lt ) * lt;

            /* 1-gamma_0 */
            /*-----------*/

            int mu = 0;
            for(int c=0;c<3;c++) {
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][y][xp/2][0][c][0]
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][3][c][1]);//PSI(1,c,3) );


              tmp_tst[0][c][1] = psi_p_f_tst[t][z][y][xp/2][0][c][1]//PSI(1,c,0)
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][3][c][0]);//PSI(0,c,3) );


              tmp_tst[1][c][0] = psi_p_f_tst[t][z][y][xp/2][1][c][0]//PSI(0,c,1)
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][2][c][1]);//PSI(1,c,2) );


              tmp_tst[1][c][1] = psi_p_f_tst[t][z][y][xp/2][1][c][1]//PSI(1,c,1)
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][2][c][0]);//PSI(0,c,2) );


              tmp_tst[2][c][0] = psi_p_f_tst[t][z][y][xp/2][2][c][0]//PSI(0,c,2)
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][1][c][1]);//PSI(1,c,1) );


              tmp_tst[2][c][1] = psi_p_f_tst[t][z][y][xp/2][2][c][1]//PSI(1,c,2)
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][1][c][0]);//PSI(0,c,1) );


              tmp_tst[3][c][0] = psi_p_f_tst[t][z][y][xp/2][3][c][0]//PSI(0,c,3)
                - sdag * (  psi_p_f_tst[t][z][y][xp/2][0][c][1]);//PSI(1,c,0) );


              tmp_tst[3][c][1] = psi_p_f_tst[t][z][y][xp/2][3][c][1]//PSI(1,c,3)
                + sdag * ( psi_p_f_tst[t][z][y][xp/2][0][c][0]);//PSI(0,c,0) );
            }

            /* multiply by U_mu */
            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] =  (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[1 + c*2 + s*3*2]= (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1-gamma_1 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //tmp[0 + c*2 + 0*2*3]
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][yp][x/2][0][c][0]//PSI(0,c,0)
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][3][c][0]);//PSI(0,c,3) );

              //tmp[1 + c*2 + 0*2*3]
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][yp][x/2][0][c][1]//PSI(1,c,0)
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][3][c][1]);//PSI(1,c,3) );

              //tmp[0 + c*2 + 1*2*3]
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][yp][x/2][1][c][0]//PSI(0,c,1)
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][2][c][0]);//PSI(0,c,2) );

              //tmp[1 + c*2 + 1*2*3]
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][yp][x/2][1][c][1]//PSI(1,c,1)
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][2][c][1]);//PSI(1,c,2) );

              //tmp[0 + c*2 + 2*2*3]
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][yp][x/2][2][c][0]//PSI(0,c,2)
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][1][c][0]);//PSI(0,c,1) );

              //tmp[1 + c*2 + 2*2*3]
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][yp][x/2][2][c][1]//PSI(1,c,2)
                - sdag * (  psi_p_f_tst[t][z][yp][x/2][1][c][1]);//PSI(1,c,1) );

              //tmp[0 + c*2 + 3*2*3]
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][yp][x/2][3][c][0]//PSI(0,c,3)
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][0][c][0]);//PSI(0,c,0) );

              //tmp[1 + c*2 + 3*2*3]
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][yp][x/2][3][c][1]//PSI(1,c,3)
                - sdag * ( -psi_p_f_tst[t][z][yp][x/2][0][c][1]);//PSI(1,c,0) );
            }

            /* multiply by U_mu */
            mu = 1;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1-gamma_2 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][zp][y][x/2][0][c][0]//PSI(0,c,0)
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][zp][y][x/2][0][c][1]//PSI(1,c,0)
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][zp][y][x/2][1][c][0]//PSI(0,c,1)
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][zp][y][x/2][1][c][1]//PSI(1,c,1)
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][zp][y][x/2][2][c][0]//PSI(0,c,2)
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][0][c][1]);//PSI(1,c,0) );

              // TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][zp][y][x/2][2][c][1]//PSI(1,c,2)
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][zp][y][x/2][3][c][0]//PSI(0,c,3)
                - sdag * ( -psi_p_f_tst[t][zp][y][x/2][1][c][1]);//PSI(1,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][zp][y][x/2][3][c][1]//PSI(1,c,3)
                - sdag * (  psi_p_f_tst[t][zp][y][x/2][1][c][0]);//PSI(0,c,1) );
            }
            /* multiply by U_mu */
            mu = 2;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1]+= (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );	

              }
            }

            /* 1-gamma_3 */
            /*-----------*/


            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[tp][z][y][x/2][0][c][0]//PSI(0,c,0)
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[tp][z][y][x/2][0][c][1]//PSI(1,c,0)
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[tp][z][y][x/2][1][c][0]///PSI(0,c,1)
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[tp][z][y][x/2][1][c][1]//PSI(1,c,1)
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[tp][z][y][x/2][2][c][0]//PSI(0,c,2)
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[tp][z][y][x/2][2][c][1]//PSI(1,c,2)
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][0][c][1]);//PSI(1,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[tp][z][y][x/2][3][c][0]//PSI(0,c,3)
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][1][c][0]);//PSI(0,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[tp][z][y][x/2][3][c][1]//PSI(1,c,3)
                - sdag * (  psi_p_f_tst[tp][z][y][x/2][1][c][1]);//PSI(1,c,1) );
            }

            /* multiply by U_mu */
            mu = 3;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    -  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][0][c][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][1][c][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cbn][t][z][y][x/2][mu][2][c][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1+gamma_0 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][y][xm/2][0][c][0]//PSI(0,c,0)
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][3][c][1]);//PSI(1,c,3) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][y][xm/2][0][c][1]//PSI(1,c,0)
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][3][c][0]);//PSI(0,c,3) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][y][xm/2][1][c][0]//PSI(0,c,1)
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][2][c][1]);//PSI(1,c,2) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][y][xm/2][1][c][1]//PSI(1,c,1)
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][2][c][0]);//PSI(0,c,2) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][y][xm/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][1][c][1]);//PSI(1,c,1) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][y][xm/2][2][c][1]//PSI(1,c,2)
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][1][c][0]);//PSI(0,c,1) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][y][xm/2][3][c][0]//PSI(0,c,3)
                + sdag * (  psi_p_f_tst[t][z][y][xm/2][0][c][1]);//PSI(1,c,0) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][y][xm/2][3][c][1]//PSI(1,c,3)
                + sdag * ( -psi_p_f_tst[t][z][y][xm/2][0][c][0]);//PSI(0,c,0) );
            }
            /* multiply by U_mu */
            mu = 0;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][t][z][y][xm/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );       		
              }
            }

            /* 1+gamma_1 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][z][ym][x/2][0][c][0]//PSI(0,c,0)
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][z][ym][x/2][0][c][1]//PSI(1,c,0)
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][z][ym][x/2][1][c][0]//PSI(0,c,1)
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][z][ym][x/2][1][c][1]//PSI(1,c,1)
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][z][ym][x/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][1][c][0]);//PSI(0,c,1) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][z][ym][x/2][2][c][1]//PSI(1,c,2)
                + sdag * (  psi_p_f_tst[t][z][ym][x/2][1][c][1]);//PSI(1,c,1) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][z][ym][x/2][3][c][0]//PSI(0,c,3)
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][z][ym][x/2][3][c][1]//PSI(1,c,3)
                + sdag * ( -psi_p_f_tst[t][z][ym][x/2][0][c][1]);//PSI(1,c,0) );
            }

            /* multiply by U_mu */
            mu = 1;

            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][t][z][ym][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1+gamma_2 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[t][zm][y][x/2][0][c][0]//PSI(0,c,0)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[t][zm][y][x/2][0][c][1]//PSI(1,c,0)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[t][zm][y][x/2][1][c][0]//PSI(0,c,1)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[t][zm][y][x/2][1][c][1]//PSI(1,c,1)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[t][zm][y][x/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][0][c][1]);//PSI(1,c,0) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[t][zm][y][x/2][2][c][1]//PSI(1,c,2)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[t][zm][y][x/2][3][c][0]//PSI(0,c,3)
                + sdag * ( -psi_p_f_tst[t][zm][y][x/2][1][c][1]);//PSI(1,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[t][zm][y][x/2][3][c][1]//PSI(1,c,3)
                + sdag * (  psi_p_f_tst[t][zm][y][x/2][1][c][0]);//PSI(0,c,1) );
            }
            /* multiply by U_mu */

            mu = 2;


            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][t][zm][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );
              }
            }

            /* 1+gamma_3 */
            /*-----------*/

            for(int c=0;c<3;c++) {
              //TMP(0,c,0)
              tmp_tst[0][c][0] = psi_p_f_tst[tm][z][y][x/2][0][c][0]//PSI(0,c,0)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][2][c][0]);//PSI(0,c,2) );

              //TMP(1,c,0)
              tmp_tst[0][c][1] = psi_p_f_tst[tm][z][y][x/2][0][c][1]//PSI(1,c,0)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][2][c][1]);//PSI(1,c,2) );

              //TMP(0,c,1)
              tmp_tst[1][c][0] = psi_p_f_tst[tm][z][y][x/2][1][c][0]//PSI(0,c,1)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][3][c][0]);//PSI(0,c,3) );

              //TMP(1,c,1)
              tmp_tst[1][c][1] = psi_p_f_tst[tm][z][y][x/2][1][c][1]//PSI(1,c,1)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][3][c][1]);//PSI(1,c,3) );

              //TMP(0,c,2)
              tmp_tst[2][c][0] = psi_p_f_tst[tm][z][y][x/2][2][c][0]//PSI(0,c,2)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][0][c][0]);//PSI(0,c,0) );

              //TMP(1,c,2)
              tmp_tst[2][c][1] = psi_p_f_tst[tm][z][y][x/2][2][c][1]//PSI(1,c,2)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][0][c][1]);//PSI(1,c,0) );

              //TMP(0,c,3)
              tmp_tst[3][c][0] = psi_p_f_tst[tm][z][y][x/2][3][c][0]//PSI(0,c,3)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][1][c][0]);//PSI(0,c,1) );

              //TMP(1,c,3)
              tmp_tst[3][c][1] = psi_p_f_tst[tm][z][y][x/2][3][c][1]//PSI(1,c,3)
                + sdag * (  psi_p_f_tst[tm][z][y][x/2][1][c][1]);//PSI(1,c,1) );
            }

            /* multiply by U_mu */
            mu = 3;
            for(int s=0;s<4;s++) {
              for(int c=0;c<3;c++) {
                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0]
                chi_tmp[c*2 + s*3*2] += (  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][0]//TMP(0,2,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][1]);////TMP(1,2,s) );

                //chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1]
                chi_tmp[c*2 + s*3*2 +1] += (   u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][0]
                    * tmp_tst[s][0][1]//TMP(1,0,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][0]
                    * tmp_tst[s][1][1]//TMP(1,1,s)

                    +  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][0]
                    * tmp_tst[s][2][1]//TMP(1,2,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][0][1]
                    * tmp_tst[s][0][0]//TMP(0,0,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][1][1]
                    * tmp_tst[s][1][0]//TMP(0,1,s)

                    -  u_p_f_tst[cb][tm][z][y][x/2][mu][c][2][1]
                    * tmp_tst[s][2][0]);//TMP(0,2,s) );

                chi_p[t-1][z-1][y-1][(x-2)/2][s][c][0] = chi_tmp[c*2 + s*3*2];
                chi_p[t-1][z-1][y-1][(x-2)/2][s][c][1] = chi_tmp[c*2 + s*3*2 +1];
              }
            }
          } // end if parity == cbn
        } // end t loop
      } // end z loop
    } // end y loop
  } // end x loop
}

void dwf_dslash_4()
{
  int cb = 0;

  float (*frm_md_out)[LZ][LY][LX/2][4][3][2] =
    (float (*)[LZ][LY][LX/2][4][3][2]) malloc(sizeof(float)*LT*LZ*LY*LX/2*4*3*2);

  float (*frm_md_out_off)[LZ][LY][LX/2][4][3][2] =
    (float (*)[LZ][LY][LX/2][4][3][2]) malloc(sizeof(float)*LT*LZ*LY*LX/2*4*3*2);

  float (*g_md_field)[LT+1][LZ+1][LY+1][(LX/2)+1][4][3][3][2] =
    (float (*)[LT+1][LZ+1][LY+1][(LX/2)+1][4][3][3][2]) malloc(sizeof(float)*2*(LT+1)*(LZ+1)*(LY+1)*((LX/2)+1)*4*3*3*2);

  float (*frm_md_in)[LZ+2][LY+2][(LX/2)+2][4][3][2] =
    (float (*)[LZ+2][LY+2][(LX/2)+2][4][3][2]) malloc(sizeof(float)*(LT+2)*(LZ+2)*(LY+2)*((LX/2)+2)*4*3*2);


#pragma omp parallel for collapse(2)
  for(int t=0; t<LT+1; t++) {
    for(int z=0; z<LZ+1; z++) {
      for(int y=0; y<LY+1; y++) {
        for(int x=0; x<LX/2+1; x++) {
          for(int c1=0; c1<2; c1++) {
            for(int mu=0;mu<4;mu++) {
              for(int s=0;s<3;s++) {
                for(int c=0;c<3;c++) {
                  for(int r=0;r<2;r++) {
                    g_md_field[c1][t][z][y][x][mu][s][c][r] = 1;//1.0/(x+1);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

#pragma omp parallel for collapse(2)
  for(int t=0; t<LT+2; t++) {
    for(int z=0; z<LZ+2; z++) {
      for(int y=0; y<LY+2; y++) {
        for(int x=0; x<LX/2+2; x++) {
          for(int s=0;s<4;s++) {
            for(int c=0;c<3;c++) {
              for(int r=0;r<2;r++) {
                frm_md_in[t][z][y][x][s][c][r] = 1;//1.0/(y+1);
              }
            }
          }
        }
      }
    }
  }

#pragma omp target enter data \
  map(alloc:frm_md_out_off[0:LT][0:LZ][0:LY][0:LX/2][0:4][0:3][0:2]) \
  map(to:g_md_field[0:2][0:LT+1][0:LZ+1][0:LY+1][0:LX/2+1][0:4][0:3][0:3][0:2]) \
  map(to:frm_md_in[0:LT+2][0:LZ+2][0:LY+2][0:LX/2+2][0:4][0:3][0:2])
  {}
  //print_in(g_md_field);
  double duration_omp;
  flush_llc();
  reset_timer();
  start_timer();
  wilson_dslash_omp(frm_md_out,
      g_md_field,
      frm_md_in,
      cb
      );
  duration_omp = stop_timer();
  fprintf(fp, "CPU,,0,0,%d,%d,%d,%d,%.0lf\n", LX,LY,LZ,LT, duration_omp);
  fflush(fp);

  mem_to = sizeof(float)*2*(LT+1)*(LZ+1)*(LY+1)*(LX/2+1)*4*3*3*2 +
    sizeof(float)*(LT+2)*(LZ+2)*(LY+2)*(LX/2+2)*4*3*2;
  mem_from = sizeof(float)*(LT)*(LZ)*(LY)*(LX/2)*4*3*2;

  //print(frm_md_out);

#pragma omp parallel for
  for(int dev=0; dev<num_dev; dev++)
    wilson_dslash(frm_md_out_off, g_md_field, frm_md_in, cb, dev);
#ifdef COMPARE
  for(int i=0; i<num_dev; i++) {
#pragma omp target exit data \
    map(from:frm_md_out_off[0:LT][0:LZ][0:LY][0:LX/2][0:4][0:3][0:2]) \
    device(i) 
    {}
  }
  compare(frm_md_out, frm_md_out_off);
  for(int i=0; i<num_dev; i++) {
#pragma omp target enter data \
    map(to:frm_md_out_off[0:LT][0:LZ][0:LY][0:LX/2][0:4][0:3][0:2]) \
    device(i) 
    {}
  }
#endif

#pragma omp parallel for
  for(int dev=0; dev<num_dev; dev++)
    wilson_dslash_collapse(frm_md_out_off, g_md_field, frm_md_in, cb, dev);
#ifdef COMPARE
  for(int i=0; i<num_dev; i++) {
#pragma omp target exit data \
    map(from:frm_md_out_off[0:LT][0:LZ][0:LY][0:LX/2][0:4][0:3][0:2]) \
    device(i) 
    {}
  }
  compare(frm_md_out, frm_md_out_off);
  for(int i=0; i<num_dev; i++) {
#pragma omp target enter data \
    map(to:frm_md_out_off[0:LT][0:LZ][0:LY][0:LX/2][0:4][0:3][0:2]) \
    device(i) 
    {}
  }
#endif

#pragma omp parallel for 
  for(int dev=0; dev<num_dev; dev++)
    wilson_dslash_split(frm_md_out_off, g_md_field, frm_md_in, cb, dev);
#ifdef COMPARE
  for(int i=0; i<num_dev; i++) {
#pragma omp target exit data \
    map(from:frm_md_out_off[0:LT][0:LZ][0:LY][0:LX/2][0:4][0:3][0:2]) \
    device(i) 
    {}
  }
  compare(frm_md_out, frm_md_out_off);
  for(int i=0; i<num_dev; i++) {
#pragma omp target enter data \
    map(to:frm_md_out_off[0:LT][0:LZ][0:LY][0:LX/2][0:4][0:3][0:2]) \
    device(i) 
    {}
  }
#endif

#pragma omp parallel for 
  for(int dev=0; dev<num_dev; dev++)
    wilson_dslash_swap(frm_md_out_off, g_md_field, frm_md_in, cb, dev);
#ifdef COMPARE
  for(int i=0; i<num_dev; i++) {
#pragma omp target exit data \
    map(from:frm_md_out_off[0:LT][0:LZ][0:LY][0:LX/2][0:4][0:3][0:2]) \
    device(i) 
    {}
  }
  compare(frm_md_out, frm_md_out_off);
  for(int i=0; i<num_dev; i++) {
#pragma omp target enter data \
    map(to:frm_md_out_off[0:LT][0:LZ][0:LY][0:LX/2][0:4][0:3][0:2]) \
    device(i) 
    {}
  }
#endif

#pragma omp parallel for 
  for(int dev=0; dev<num_dev; dev++)
    wilson_dslash_collapse_swap(frm_md_out_off, g_md_field, frm_md_in, cb, dev);
#ifdef COMPARE
  for(int i=0; i<num_dev; i++) {
#pragma omp target exit data \
    map(from:frm_md_out_off[0:LT][0:LZ][0:LY][0:LX/2][0:4][0:3][0:2]) \
    device(i) 
    {}
  }
  compare(frm_md_out, frm_md_out_off);
  for(int i=0; i<num_dev; i++) {
#pragma omp target enter data \
    map(to:frm_md_out_off[0:LT][0:LZ][0:LY][0:LX/2][0:4][0:3][0:2]) \
    device(i) 
    {}
  }
#endif

#pragma omp parallel for 
  for(int dev=0; dev<num_dev; dev++)
    wilson_dslash_split_swap(frm_md_out_off, g_md_field, frm_md_in, cb, dev);
#ifdef COMPARE
  for(int i=0; i<num_dev; i++) {
#pragma omp target exit data \
    map(from:frm_md_out_off[0:LT][0:LZ][0:LY][0:LX/2][0:4][0:3][0:2]) \
    device(i) 
    {}
  }
  compare(frm_md_out, frm_md_out_off);
  for(int i=0; i<num_dev; i++) {
#pragma omp target enter data \
    map(to:frm_md_out_off[0:LT][0:LZ][0:LY][0:LX/2][0:4][0:3][0:2]) \
    device(i) 
    {}
  }
#endif

#pragma omp parallel for
  for(int i=0; i<num_dev; i++) {
#pragma omp target exit data \
    map(from:frm_md_out_off[0:LT][0:LZ][0:LY][0:LX/2][0:4][0:3][0:2]) \
    map(delete: g_md_field[0:2][0:LT+1][0:LZ+1][0:LY+1][0:LX/2+1][0:4][0:3][0:3][0:2]) \
    map(delete: frm_md_in[0:LT+2][0:LZ+2][0:LY+2][0:LX/2+2][0:4][0:3][0:2]) \
    device(i) 
    {}
  }
}

int main(int argc, char **argv) 
{
  std::string output_file_name;
  if(argc > 1) {
    output_file_name = argv[1];
  } else {
    output_file_name = argv[0];
    output_file_name = output_file_name.substr(output_file_name.find_last_of("/\\")+1);
    output_file_name = output_file_name.substr(0, output_file_name.size() - 3);
    output_file_name = "output_" + output_file_name + "csv";
  }
  printf("%s\n", output_file_name.c_str());
  fp = fopen(output_file_name.c_str(), "w");
  num_dev = 1;//omp_get_num_devices();

  fprintf(fp, "Device,transform,mem_to,mem_from,LX,LY,LZ,LT,runtime(us)\n");
  dwf_dslash_4();
  return 0;
}
