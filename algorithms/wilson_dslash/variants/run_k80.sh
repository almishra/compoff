for ((i=4; i<=64; i+=4))
do
  echo $i;
  t=$(( 2 * i ));
  ./dslash_${i}_${i}_${i}_${t}_k80.out;
  ./dslash_${i}_${i}_${i}_${t}_k80_memcpy.out;
done
