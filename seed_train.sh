for i in {42..46};do
echo "Training with seed $i";
python new_finetune_offensive_fewshot.py hasoc-2020/task1-ml $i maml_v8 state;
done

#for i in {42..46};do
#echo "Training with seed $i";
#python new_finetune_offensive_fewshot.py hasoc-2020/task2-ta $i proto_v8 state;
#done

for i in {42..46};do
echo "Training with seed $i";
python new_finetune_offensive_fewshot.py hasoc-2020/task2-ml $i maml_v8 state;
done
