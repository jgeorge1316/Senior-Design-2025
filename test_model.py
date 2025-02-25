from ultralytics import YOLO
import time

if __name__ == "__main__":
    # Load a model
    #model = YOLO("weights/best.pt")  # Load the model
    #model = YOLO("runs/classify/train2/weights/last.pt")
    model = YOLO("single_model0.1.1.pt")
    start_time=time.perf_counter()
    results1 = model.predict("/home/landon/Senior-Design/Training/none", stream=1)

    count_total = 0
    count_correct = 0
    count_wrong = 0
    
    for result in results1:
        #{0: 'narrowleaf_cattail', 1: 'none', 2: 'phragmites', 3: 'purple_loosestrife'}
        if(result.probs.top1 == 1): #set the number to the class folder which is currently being tested
            count_correct+=1
        else:
            count_wrong+=1
        count_total+=1
    
    end_time=time.perf_counter()
    elapsed_time=end_time-start_time
    print(f"Total process Time for the folder: {elapsed_time:.4f} seconds.\n")

    print(f"Correct: {count_correct} Wrong: {count_wrong} Total: {count_total}")
    percentage_correct = count_correct/count_total
    print(f"Percentage Correct: {percentage_correct:.4f}\n")

