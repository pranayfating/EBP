Error backpropagation Algorithm:
    Idea behind backpropagation algorithm is quite simple, output of NN is evaluated against desired output. If results are not satisfactory,connection (weights) between layers are modied and process is repeated again and again until error is small enough.

Requirements:
    Matplotlib 3.0
    Python 3.6

Dataset:
    Avila Data Set(https://archive.ics.uci.edu/ml/datasets/Avila)

    The Avila data set has been extracted from 800 images of the the "Avila Bible", a giant Latin copy of the whole Bible produced during the XII century between Italy and Spain.  
    The palaeographic analysis of the  manuscript has  individuated the presence of 12 copyists. The pages written by each copyist are not equally numerous. 
    Each pattern contains 10 features and corresponds to a group of 4 consecutive rows.

    The prediction task consists in associating each pattern to one of the 12 copyists (labeled as: A, B, C, D, E, F, G, H, I, W, X, Y).
    The data have has been normalized, by using the Z-normalization method, and divided in two data sets: a training set containing 10430 samples, and a test set  containing the 10437 samples.

    Class distribution (training set)
    A: 4286
    B: 5  
    C: 103 
    D: 352 
    E: 1095 
    F: 1961 
    G: 446 
    H: 519
    I: 831
    W: 44
    X: 522 
    Y: 266

    ATTRIBUTE DESCRIPTION

    ID      Name    
    F1       intercolumnar distance 
    F2       upper margin 
    F3       lower margin 
    F4       exploitation 
    F5       row number 
    F6       modular ratio 
    F7       interlinear spacing 
    F8       weight 
    F9       peak number 
    F10     modular ratio/ interlinear spacing
    Class: A, B, C, D, E, F, G, H, I, W, X, Y


Algorithms Used :
    1)k-fold cross validation
    2)sigmoid activation function
    3)gradient descent


How to execute ?
    Place the files "EBP.py" and "avila-tr.csv" in the same directory and run "EBP.py"

Graph :
    Generated graph is the relation between "Mean squared Error" and "Epochs" and each color denotes a fold.