Few clarifications before we get into the code.
1.I have not included any requirements file for this repo because the entire repo was run on the base installation of python (which is 2.7) and due to the presence of other softwares extraneous to the project a simple pip freeze brings up requirements that are not necessary for the run. For this, what I would suggest, is start with the packages I mention here and install a package if it is missing. 
2. The list of obvious packages:
  1. numpy==1.15.4
  2. torch==0.4.0
  3. matplotlib==2.1.1
  4. gym==0.10.9
  5. -e git+https://github.com/ranok92/gym-ballenv.git@855b040af6e4eb1a1e4b8ee263d30038a02d8b33#egg=gym_ballenv
3. Once the dependencies are satisfied, things should be pretty straight forward.

Steps involved in running the trained models.
  1.Clone the repo.
  2.Follow the instructions above to satisfy the package requirements.
  3.Go into the examples folder.
  4.Run this command
   python ball_cnn_ac3.py --resume "./stored_models/ball_state3/2layer+dropout+randpos/episode_2500.pth".
  5.For running other models:
  If you look into the stored_models folder,you will find a number of pretrained models with different network architectures and          local state representations present. Running any one of the models will involve the following steps:
      a. Pass the path of the model you want to run into the --resume argument. Both relative path and absolute path will work.
      b. Now to the tricky part. As said before, experiments have been run on different local state representations, namely 5x5 and 10x10 window representation for the local information. This changes the size of the neural network and as because the models stored are the weight dictionaries and not the actual model a little change needs to be done if you want to try out results from different feature representations. So, when switching between models trained in different window sizes (5 and 10), open the 'examples/ball_cnn_ac3.py' file and change the value of the variable WINDOW in line 493 to the appropriate value (5 or 10)
 based on the result of the model you want to see. Thats it. If you face any issue/difficulty following the procedure feel free to contact me.
 
      c. Finally, 
      Folders with models trained on window size 5:
        stored_models/ball_state3/2 layer
        stored_models/ball_state3/2 layer+dropout
        stored_models/ball_state3/2 layer+dropout+randpos/
      Folders with models trained on window size 10:
        stored_models/ball_state3/2 layer+dropout+randpos/window_size_10
