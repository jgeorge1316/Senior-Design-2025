from ultralytics import YOLO
import time

if __name__ == "__main__":
    # Load the model
    model = YOLO("models/single_model0.1.1.pt")  # Load a pretrained model
    start_time=time.perf_counter()
    results = model.predict("/home/landon/Senior-Design/small_dataset/test/narrowleaf_cattail", stream=1) #you can set this to a specific image, or to a folder of images
    #print(results) #prints the ultralytics engine results structure. For classification, you only need to focus on result.names, and result.probs
    
    for result in results:
        #print(result.probs)  # This will print the Probs object

        #result.show() #shows the result image (I don't think this will be useful for the final deployment)

        print(result.names) #prints the names array
        #names array for above model: {0: 'narrowleaf_cattail', 1: 'none', 2: 'phragmites', 3: 'purple_loosestrife'}

        print(result.probs.data.tolist())  #prints the entire result.probs, which has all of the classification results

        print(result.probs.top1) #result.probs.top1 returns the index of the class with the highest probability

        print(result.names[result.probs.top1]) #using top1, and the class names array, print the class with the highest probability

        print(result.speed) #prints speed object

    #you need the end time counter to be AFTER at least the first time that you access the results, if the stream=1 in model.predict.
    #stream=1 changes the way that the predict function behaves, and makes it use way less memory, but it also makes the python interpreter start moving to the next line of code
    #if you put the end_time after any line using results, then it will work since the interpreter will wait for model.predict to finish so that it can get the results variable
    end_time=time.perf_counter()
    elapsed_time=end_time-start_time
    print(f"Total process Time for the folder: {elapsed_time:.4f} seconds.\n")
