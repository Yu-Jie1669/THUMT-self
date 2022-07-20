#!/bin/bash

. `dirname $0`/../common/vars

src=tr
tgt=en
pair=$src-$tgt
#
## Tokenise
for lang in $src $tgt; do
  cat \
    $opus_dir/SETIMES2.$tgt-$src.$lang  |
   $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang | \
   $moses_scripts/tokenizer/tokenizer.perl -a -l $lang  \
   > corpus.tok.$lang
done
#
###
#### Clean
$moses_scripts/training/clean-corpus-n.perl corpus.tok $src $tgt corpus.clean 1 $max_len corpus.retained
###
#
#### Train truecaser and truecase
for lang in $src $tgt; do
  $moses_scripts/recaser/train-truecaser.perl -model truecase-model.$lang -corpus corpus.tok.$lang
  $moses_scripts/recaser/truecase.perl < corpus.clean.$lang > corpus.tc.$lang -model truecase-model.$lang
done
#
#
  
  
# dev sets
for testset in newsdev2016 newstest2016; do
  for lang  in $src $tgt; do
    side="src"
    if [ $lang = $tgt ]; then
      side="ref"
    fi
    $moses_scripts/ems/support/input-from-sgm.perl < $dev_dir/$testset-$src$tgt-$side.$lang.sgm | \
    $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang | \
    $moses_scripts/tokenizer/tokenizer.perl -a -l $lang |  \
    $moses_scripts/recaser/truecase.perl   -model truecase-model.$lang \
    > $testset.tc.$lang
    
  done
  cp $dev_dir/$testset-$src$tgt*sgm .
  cp $dev_dir/$testset-$tgt$src*sgm .
done

## Tidy up and compress
for lang in $src $tgt; do
  gzip corpus.tc.$lang
  rm -f corpus.tok.$lang corpus.clean.$lang corpus.retained
done
tar zcvf dev.tgz news* &&  rm news*
tar zcvf true.tgz truecase-model.* && rm truecase-model*