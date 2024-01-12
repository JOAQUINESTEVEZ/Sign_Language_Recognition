# Sign Language Recognition Using LSTM Neural Networks 
// 👨‍💻Author: Joaquin Estevez Year: 2024

## ![Captura de pantalla 2024-01-10 192402](https://github.com/JOAQUINESTEVEZ/Sign_Language_Recognition/assets/105304562/665c144f-eb14-411a-b2c3-fe2248709740)




## 🌐Overview 

In this project, a model which detects hand signs in real time and accurately identifies the corresponding letters of the American Sign Language alphabet was developed, allowing users to practice and improve their sign language. 

## 🤲American Sign Language (ASL) 

Sign languages are languages that use the visual-manual modality to convey meaning, instead of spoken words. Sign languages are expressed through manual articulation in combination with non-manual markers. Sign languages are full-fledged natural languages with their own grammar and lexicon. Sign languages are not universal and are usually not mutually intelligible, although there are also similarities among different sign languages. For this model, the ASL was selected, having the following alphabet:
  
  ![alphabet_chart](https://github.com/JOAQUINESTEVEZ/Sign_Language_Recognition/assets/105304562/bd7b7974-d4b8-4742-aa26-7a0e31f5349e)

## 🤖Model 
- A LSTM (Long Short-Term Memory) Neural Networks model was choosen.
- ### Why?
  - It excels at capturing long-term dependencies, making it ideal for sequence prediction tasks.
  - Unlike traditional neural networks, LSTM incorporates feedback connections, allowing it to process entire sequences of data, not just individual data points, which is very useful to determine signs that require movement and are not static.
- A final Dense layer was created with a softmax activation function to output a probability distribution over the possible alphabet letters.
- An Adam optimizer was selected along with a Categorical Cross Entropy loss used for multiclass classification.
    - ```python
      model = Sequential()
      model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
      model.add(LSTM(128, return_sequences=True, activation='relu'))
      model.add(LSTM(64, return_sequences=False, activation='relu'))
      model.add(Dense(64, activation='relu'))
      model.add(Dense(32, activation='relu'))
      model.add(Dense(actions.shape[0], activation='softmax'))
      ```
      ```python
      model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
      ```

## 🔧Usage

1. Inside your venv, install the required packages by running:
   ```bash
   pip install -r requirements.txt
   ```
2. Go to the ASL.ipynb file.
   
3. Go to Section 11: Test in Real Time.

4. Run each cell of this section.
    - If you get an error after importing the dependencies, you may need to install extra packages
  
5. The last cell will open a camera window and the model will start making predictions.

    ## 📌Important Notes:
  
      1. Make sure your camera is properly connected, and the environment has adequate lighting for accurate hand sign detection.
      2. In case you are having issues with the camera, try changing "cap = cv2.VideoCapture(0)" by "cap = cv2.VideoCapture(1)" in the last cell of Section 11. If the issue continues, try with 2, 3, or 4:
  
          ```python
          cap = cv2.VideoCapture(0) # change the '0' to '1', '2', '3', or '4' depending on your set up
          ```

6. Perform hand signs in front of your camera to practice and test your ASL:

      ### You can use the alphabet image above to have a reference. [American Sign Language (ASL)](#american-sign-language-(asl))

7. Enjoy!

8. 🚪Press 'q' when you want to finish the program.
