./lud_omp -m 100 >> ~/projects/compoff/benchmarkdata/lud.csv
./lud_omp -m 1000 >> ~/projects/compoff/benchmarkdata/lud.csv
./lud_omp -m 2000 >> ~/projects/compoff/benchmarkdata/lud.csv
./lud_omp -m 4000 >> ~/projects/compoff/benchmarkdata/lud.csv
./lud_omp -m 6000 >> ~/projects/compoff/benchmarkdata/lud.csv
./lud_omp -m 8000 >> ~/projects/compoff/benchmarkdata/lud.csv
for i in {10000..100000..5000}
  do 
     ./lud_omp -m $i >> ~/projects/compoff/benchmarkdata/lud.csv
 done
#100, 1000, 2000, 4000, 6000, 8000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000