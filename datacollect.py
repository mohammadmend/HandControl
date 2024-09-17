import cv2 as cv
import mediapipe as mp 
import csv
def collectdata():
 
    mphand=mp.solutions.hands
    hands=mphand.Hands()
    mpdraw=mp.solutions.drawing_utils
    with open('hand_Data_.csv',mode='w',newline='')as file:
        writer=csv.writer(file)
        writer.writerow(['gesture'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)])
        cap=cv.VideoCapture(1)
        while True:
            ret,frame=cap.read()
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            #cv.imshow('Hand Gesture Capture', frame)
            if result.multi_hand_landmarks:
                for hand_index,hand_landmarks in enumerate(result.multi_hand_landmarks):
                        #print("Index",hand_index)
                        handedness = result.multi_handedness[hand_index].classification[0].label

                        mpdraw.draw_landmarks(frame, hand_landmarks, mphand.HAND_CONNECTIONS)
                        land=[]
                        vis=True
                        # xcord=[lm.x for lm in hand_landmarks.landmark]
                        # ycord=[lm.y for lm in hand_landmarks.landmark]
                        # minx=min(xcord)
                        # miny=min(ycord)
                        # maxx=max(xcord)
                        # maxy=max(ycord)

                        for id,lm in enumerate(hand_landmarks.landmark):
                            land.extend([lm.x,lm.y,lm.z])
                            key = cv.waitKey(1) & 0xFF
                            if key == ord("n"):
                                print("what gesture MO")
                                label=input()
                                writer.writerow([label,handedness]+land)
                                
                            #print("Hand",id)
                        
                            # if not (0.5 <= lm.x <= 1 and 0.5 <= lm.y <= 1):
                            #     print("ge",lm.visibility)
                            # #if lm.visibility<0.5:
                            #     vis=False
                            #     print("bruh")
                            #     break
                        # if not (0<maxx<=1 and 0<maxy<=1 and 0<minx<=1 and 0<miny<=1):
                        #     vis=False
                        
                        
                        
            cv.imshow('feed',cv.flip(frame,1))
            if cv.waitKey(1)==ord("q"):break

def main():
    collectdata()
if __name__=="__main__":
    main()

