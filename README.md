# CS5242_Project2020

UFC101 leaderboard: https://paperswithcode.com/sota/action-recognition-in-videos-on-ucf101?p=i3d-lstm-a-new-model-for-human-action

Referred paper: https://arxiv.org/pdf/1705.07750.pdf

### Some Tips
    // results data(losses/scores)
    ./results/
        - fc
        - lstm
    
    // print out during process
    ./log/
        - fc
        - lstm
    
    // trained model
    ./trained/

### Some util funcs
    // utils/draw.py
    // show plot
    python utils/draw.py ./results/fc/training_losses_fc.npy ./results/fc/training_scores_fc.npy

    // ./eval.py
    // evaluate model
    python eval.py ./trained/fc_devtrain.pkl