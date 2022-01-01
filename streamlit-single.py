import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import pandas as pd

def main():
  data=pd.read_csv(r'var 201221.csv')

  data["DERMAGA"]=data["DERMAGA"].astype('category')
  data["JENIS_KAPAL"]=data["JENIS_KAPAL"].astype('category')
  data["DELAY"]=data["DELAY"].astype('category')
  data["WAG"]=data["WAG"].astype('category')
  data["BAD_WEATHER"]=data["BAD_WEATHER"].astype('category')

  

  arr = data.values
  X = arr[:, 0:7]
  Y = arr[:, 7]

  X_train, X_test, Y_train, Y_test=train_test_split(X, Y,test_size=.2)

  knn=KNeighborsClassifier(n_neighbors=2)
  knn.fit(X_train,Y_train)
  Y_PredKNN=knn.predict(X_test)

  #Saving the Model
  pickle_out = open("knn.pkl", "wb") 
  pickle.dump(knn, pickle_out) 
  pickle_out.close()

  pickle_in = open('knn.pkl', 'rb')
  classifier = pickle.load(pickle_in)

 
  st.header('Prediksi Delay Keberangkatan Kapal')
  DERMAGA = st.number_input("Dermaga Sandar:")
  JENIS_KAPAL = st.selectbox("Jenis Kapal:", ("Feeder", "Direct"))
  PALKA =  st.number_input("Jumlah Palka (Unit):")
  BD = st.number_input("Lama Waktu Breakdown (Menit):")
  SHIFTING_YARD = st.number_input("Jumlah Shifting Yard (Box):")
  WAG = st.selectbox("Status WAG:", ("Ada", "Tidak Ada"))
  BAD_WEATHER = st.selectbox("Bad Weather:", ("Ada", "Tidak Ada"))
  submit = st.button('Predict')
  if submit:
    if BAD_WEATHER == "Ada":
      BAD_WEATHER = 1
    else:
      BAD_WEATHER = 0
    if WAG == "Ada":
      WAG = 1
    else:
      WAG = 0
    if JENIS_KAPAL == "Direct":
      JENIS_KAPAL = 1
    else:
      JENIS_KAPAL = 0
      
    prediction = classifier.predict([[DERMAGA, JENIS_KAPAL, PALKA, BD, SHIFTING_YARD, WAG, BAD_WEATHER]])
    st.write(prediction)
    if prediction == 0:
        st.write('KAPAL TIDAK MENGALAMI DELAY KEBERANGKATAN')
    elif prediction == 1:
        st.write('KAPAL MENGALAMI DELAY KURANG DARI 4 JAM')
    else:
        st.write('KAPAL MENGALAMI DELAY LEBIH DARI 4 JAM')

if __name__ == '__main__':
  main()
