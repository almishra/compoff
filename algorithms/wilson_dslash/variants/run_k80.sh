for ((i=4; i<=64; i+=4))
do
  echo $i;
  t=$(( 2 * i ));
  if [ ! -f output_dslash_${i}_${i}_${i}_${t}_k80.csv ]
  then
    ./dslash_${i}_${i}_${i}_${t}_k80.out;
  fi
  if [ ! -f output_dslash_${i}_${i}_${i}_${t}_k80.csv ]
  then
    ./dslash_${i}_${i}_${i}_${t}_k80_memcpy.out;
  fi
done
