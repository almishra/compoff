for ((i=4; i<=64; i+=4))
do
  echo $i;
  t=$(( 2 * i ));
  if [ ! -f output_dslash_${i}_${i}_${i}_${t}_v100.csv ]
  then
    ./dslash_${i}_${i}_${i}_${t}_v100.out;
  fi
  if [ ! -f output_dslash_${i}_${i}_${i}_${t}_v100_memcpy.csv ]
  then
    ./dslash_${i}_${i}_${i}_${t}_v100_memcpy.out;
  fi
done
