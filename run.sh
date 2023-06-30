#with best threshold for each model
#retro will print the error with details for a model

python ToxiSpanSE.py --algo roberta --tokenizer roberta --threshold 0.12 --retro
python ToxiSpanSE.py --algo bert --tokenizer bert --threshold 0.15 --retro
python ToxiSpanSE.py --algo xlnet --tokenizer xlnet --threshold 0.10
python ToxiSpanSE.py --algo albert --tokenizer albert --threshold 0.11
python ToxiSpanSE.py --algo dbert --tokenizer dbert --threshold 0.17


########the following command are for predicting on validation set#####

#python ToxiSpanSE.py --algo xlnet --tokenizer xlnet --vary
#python ToxiSpanSE.py --algo albert --tokenizer albert --vary
#python ToxiSpanSE.py --algo dbert --tokenizer dbert --vary
#python ToxiSpanSE.py --algo roberta --tokenizer roberta --vary
#python ToxiSpanSE.py --algo bert --tokenizer bert --vary


#########to run the naive model use following command#############
##python naive_algorithm.py


##to calculate the krippendorff alpha, run
#python krippendorff_alpha.py

###to agrregate the results from all threshold in validation set and find the optimal threshold using graph
#python aggregate_results.py
