echo "------------------------------";
echo "Base task 1";
for i in {42..46};do
echo "Training with seed $i";
python new_finetune_offensive_full.py hasoc-2020/task1-ml $i base state;
done

echo "------------------------------";
echo "Base task 2 ta";
for i in {42..46};do
echo "Training with seed $i";
python new_finetune_offensive_full.py hasoc-2020/task2-ta $i base state;
done


echo "------------------------------";
echo "Base task 2 ml";
for i in {42..46};do
echo "Training with seed $i";
python new_finetune_offensive_full.py hasoc-2020/task2-ml $i base state;
done

echo "------------------------------";
echo "MAML 7 task 1";
for i in {42..46};do
echo "Training with seed $i";
python new_finetune_offensive_full.py hasoc-2020/task1-ml $i maml_7 state;
done

echo "------------------------------";
echo "MAML 7 task 2 ta";
for i in {42..46};do
echo "Training with seed $i";
python new_finetune_offensive_full.py hasoc-2020/task2-ta $i maml_7 state;
done

echo "------------------------------";
echo "MAML 7 task 2 ml";
for i in {42..46};do
echo "Training with seed $i";
python new_finetune_offensive_full.py hasoc-2020/task2-ml $i maml_7 state;
done
