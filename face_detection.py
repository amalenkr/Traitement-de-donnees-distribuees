import time
import os
import glob
import cv2
import multiprocessing as mp
from multiprocessing import Process
from threading import Thread

cpus = mp.cpu_count()

haar_cascade_face = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

def face_detection(image_path):
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    copy = image.copy()
    image_gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    faces_rects = haar_cascade_face.detectMultiScale(image_gray, scaleFactor = 1.2, minNeighbors = 5);
    if len(faces_rects) !=0:
        for (x,y,w,h) in faces_rects:
            face_rect = cv2.rectangle(copy, (x, y), (x+w, y+h), (0, 255, 0), 5)
        cv2.imwrite("Data/detected_faces/" + image_name, face_rect)

def main():
    images = list(glob.iglob("Data/covers/*.jpg"))
    images.sort()
    
    start = time.time()
    for i in images:
        face_detection(i)
    end = time.time()
    print("Series computation: {} secs\n".format(end - start))
    
    start = time.time()
    threads = []
    for i in images:
        t = Thread(target=face_detection, args=(i,))
        threads.append(t)
        t.start()
     
    for t in threads: t.join()
    end = time.time()
    print("Multithreading computation: {} secs\n".format(end - start))
    
    start = time.time()
    with mp.Pool(cpus) as p:
        p.map(face_detection, images)
        p.close()
        p.join()
    end = time.time()
    print("Multiprocessing computation (with Pool): {} secs\n".format(end - start))
    
    start = time.time()
    processes = []
    for i in images:
        p = Process(target=face_detection, args=(i,))
        processes.append(p)
        p.start()
      
    for p in processes: p.join()
    end = time.time()
    print("Multiprocessing computation (with Process): {} secs\n".format(end - start))
    
if __name__ == '__main__':
    # Better protect your main function when you use multiprocessing
    main()