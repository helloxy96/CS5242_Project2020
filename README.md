# CS5242_Project2020

UFC101 leaderboard: https://paperswithcode.com/sota/action-recognition-in-videos-on-ucf101?p=i3d-lstm-a-new-model-for-human-action

Referred paper: https://arxiv.org/pdf/1705.07750.pdf

### command
    // eg. train 'Tea' Task
    python train_tasks_by_conv_1_to_1.py 'Tea'
    python train_tasks_by_conv_3_to_1.py 'Tea'
    
    // test and get result csv in 'results/final/'
    python test_tasks_by_models.py
    

### Structure
    // results data(final csv file)
    ./results/
        - final
        
    // trained model
    ./trained/
    
    // utils directory
    ./utils/

