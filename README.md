# ANNalog

contains the api for ANNalog package, datasets are not included here. 

**how to install**

cd annalog_package

pip install .

**Generation guides**

There is one generation script named ‘colab_generation_tool.py’ in the ANNalog directory, together with the installation yml file and the actual annalog package. Here is how this works:

--vocab_path				The vocabulary file, this is one pickle file provided together with the model prior under ckpt_and_vocab directory, here user need to input the actual path to the pickle file

--model_checkpoint_path		The path to the generation model, which is the Lev_extended.pt file, and this is under the directory of ckpt_and_vocab, together with the vocab pickle file. 

--generation_method		    Three generation methods available here, C-beam (classic beam search), BF-beam (best first beam search), sample (multinomial sampling).

--temperature			    Temperature setting for multinomial sampling method, range between 1.0 and 2.0 is recommended. 

--prefix				    Fix the first few number of characters in the SMILES string unchanged during generation, for instance, fixing “c1ccccc1” while generating med-chem similar molecules around “c1ccccc1F”, keep in mind the prefixed characters should be at the beginning of the input SMILES.

--filter_invalid			Make sure that during generation process, all generated SMILES are valid SMILES.

--generation_number		    How many SMILES strings should be generated.

--input_SMILES			    The input SMILES strings.

--exploration_method		Choose among ‘normal’ (direct generation using original input SMILES), ‘variants’ (generation using different SMILES representing the same molecule) and ‘recursive’ (generation by inputting generated SMILES).

--variant_number			Only input this when choosing ‘variants’ as exploration method, how many different SMILES variants representing the same molecule should be generated while generation.
--loops				Only input this when choosing ‘recursive’ as exploration method, how many loops of generation should be conducted. 
