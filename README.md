# PPC_EC2

# ROUTES 

/predict - POST 
receives an image (example in demo), and returns a number that corresponds to the class of the object 

/train - PUT 
trains the network with the latest data 

/add_data - PUT, Params : label
receives an image (just like in predict) and a param in url called label which is an int corresponding to the real label 

/quizz_data - GET 
receives a response with 3 images, 2 of them having a label, 1 being unlabeled
