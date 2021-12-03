summit="1,0,0,0,Clang,Summit,4,16,2,3800,32,32,512,10240,NVIDIA Tesla V100-SXM2-16GB,300,512,877,900,16160,1312,5120,7,32,64,2048,32,65536,65536,255,1024,64,1024,96";
seawulf="1,0,0,0,Clang,Seawulf,1,14,2,1200,32,32,256,35840,NVIDIA Tesla K80-PCIe-12GB,16,128,2505,240,11441,562,2496,3.7,32,64,2048,16,65536,65536,255,1024,192,341,16";
ookami="";
exxact="";

CLUSTER="summit";
INIT=${summit};
OUTPUT="output.csv";
while getopts ":d:o:" arg; do
  case $arg in
    d) CLUSTER=$OPTARG;;
    o) OUTPUT=$OPTARG;;
  esac
done
case $CLUSTER in
  "summit") INIT=${summit};;
  "seawulf") INIT=${seawulf};;
  "ookami") INIT=${ookami};;
  "exxact") INIT=${exxact};;
esac
echo $CLUSTER;
echo $INIT;
echo $OUTPUT;

echo "kernel,clang,gcc,nvc,intel,Compiler,Cluster,thread_per_core,core_per_socket,num_sockets,cpu_clock,l1i,l1d,l2,l3,gpu_name,connector_bandwidth,num_memory_bus,memory_clock,memory_bandwidth,memory_total,sm_clock,num_cores,compute_capability,threads_per_wrap,max_wraps_per_sm,max_threads_per_sm,max_thread_blocks_per_sm,max_32-bit_registers_per_sm,max_registers_per_block,max_registers_per_thread,max_thread_block_size,fp32_cores_per_sm,sm_registers_per_fp32_cores,shared_memory_size_per_sm,collapse1,collapse2,collapse3,collapse4,static,dynamic,guided,Iter,VarDecl,refExpr,intLiteral,floatLiteral,mem_to,mem_from,add_sub_int,add_sub_float,mul_int,mul_float,div_int,div_float,bit_rel_logic_int,bit_rel_logic_float,rem_int,assign_int,assign_float,Iter_log,VarDecl_log,refExpr_log,intLiteral_log,floatLiteral_log,mem_to_log,mem_from_log,add_sub_int_log,add_sub_float_log,mul_int_log,mul_float_log,div_int_log,div_float_log,bit_rel_logic_int_log,bit_rel_logic_float_log,rem_int_log,assign_int_log,assign_float_log,runtime" > $OUTPUT
for i in `ls ${CLUSTER}/output_mv_clang_*`;
do 
  var=($(echo ${i} | awk -F'_' '{print $4 " " $5 " " $6 " " $7 " " $8 " " $9}'));
  output=($(clang++ -fopenmp -Xclang -load -Xclang /gpfs/projects/ChapmanGroup/alok/git/cost-model/cn-mem/opt/llvm/lib/InstructionCount.so -Xclang -plugin -Xclang -inst-count matrix_vector.cpp -c -DLA=${var[0]} -DLB=${var[1]} -DLC=${var[2]} -DLD=${var[3]} -DLE=${var[4]} -DCOUNT=${var[5]} 2> /dev/null));
  for line in ${output[*]};
  do
    if [[ "$line" == "main"* ]];
    then
      continue;
    fi
    kernel_name="${CLUSTER}_"$(echo $line | awk -F',' '{print $1}');
    collapse=($(echo $line | awk -F"," '{print $2, $3, $4, $5}'));
    schedule=($(echo $line | awk -F"," '{print $6, $7, $8}'));
    line=$(echo "-- $line" | cut -d',' -f2-);

    if [ "${collapse[0]}" = "1" ]; then coll=1;
    elif [ "${collapse[1]}" = "1" ]; then coll=2;
    elif [ "${collapse[2]}" = "1" ]; then coll=3;
    elif [ "${collapse[3]}" = "1" ]; then coll=4;
    fi

    if [ "${schedule[0]}" = "1" ]; then sched="static";
    elif [ "${schedule[1]}" = "1" ]; then sched="dynamic";
    elif [ "${schedule[2]}" = "1" ]; then sched="guided";
    fi

    kernels=($(grep $sched $i | grep GPU0));
    for kern in ${kernels[*]}
    do
      temp=($(echo "${kern}" | awk -F"," '{print $2, $3, $14}'))
      if [ "${temp[0]}" = "${sched}" ] && [ "${temp[1]}" = "$coll" ]; 
      then 
        echo -ne "$kernel_name,";
        echo -ne "${INIT},";
        echo -ne $line;
        echo $line | awk -F"," '{for(i=8; i<NF;i++) { if($i==0) {printf "0," } else {printf "%lf,", log($i)/log(10)} } }'
        echo "${temp[2]}"; 
      fi
    done
  done
done >> $OUTPUT
