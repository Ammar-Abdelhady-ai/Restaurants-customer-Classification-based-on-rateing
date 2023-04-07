import streamlit as sl
import pandas as pd
import numpy as np
import joblib

path = r"D:\other\Epsilon\Epsilon Training\project 1\my solution_project 1\save files"

locations = joblib.load(f"{path}\\location_unique.h5")
rest_types = joblib.load(f"{path}\\rest_type_unique.h5")
listed_in_type = joblib.load(f"{path}\\listed_in(type).h5")
label_to_listed_type = joblib.load(f"{path}\\label_to_listed_type.h5")
cuisine_types = joblib.load(f"{path}\\cuisine_types.h5")
cuisines_selection = joblib.load(f"{path}\\cuisines_selection.h5")
column = joblib.load(f"{path}\\input_columns.h5")


cost = joblib.load(f"{path}\\cost.h5")
city = joblib.load(f"{path}\\city.h5")


tfidf = joblib.load(f"{path}\\tfidf.h5")
pipeline = joblib.load(f"{path}\\pipeline.h5")
model = joblib.load(f"{path}\\model.h5")



def convert_to_binary(label):
    if label == "Yes":
        return 1
    else:
        return 0
    
def Change(param, value):
    return print(f"{param} is change to :  {value}")


sl.markdown("""

<style>

    .css-9s5bis.edgvbvh3{
        display:none;
    }
    
    .css-1q1n0ol.egzxvld0
    {
        display: none;    
    }

<style>
            """, unsafe_allow_html=True)



sl.markdown("<h1>Restaurants Form</h1", unsafe_allow_html=True)


#-----------------------------------------------------------------------------
with sl.form("Form", clear_on_submit=True):
    #-------------------------------frist row-------------------------------
    col1, col2 = sl.columns(2)
    
    online_order = col1.selectbox("Do you have Online Order ?", options=("Yes", "No"))
    Change("Online order ", online_order)
    
    book_table = col2.selectbox("Do you have Book table ?", options=("Yes", "No"))
    Change("Book Tamble", book_table)
    
    # -----------------------------------------------------

    rest_type  = sl.selectbox("Inter your Restaurants Type", options=tuple(rest_types))
    Change("Restaurants Type", rest_type)
    
    categories = sl.selectbox("What is your Restaurant Categories ?", options=tuple(listed_in_type))
    Change("Restaurant Categories", categories)
    
    cost_of_tow = sl.selectbox("What is your Average Cost for two people ? (from to)", options=tuple(cost))
    Change("Average Cost for two people", cost_of_tow)
    
    # ------------------------------last row------------------------------
    loc, and_city = sl.columns(2)
    location = loc.selectbox("Inter your Location", options=tuple(locations))
    Change("location", location)
    
    loc_city = and_city.selectbox("City", options=tuple(city))
    Change("Neighborhood or City", loc_city)
    

    cuisines = sl.multiselect('Select type or types of your Cuisine:', cuisines_selection, default=["pizza", "coffee"])  
    Change("Cuisines Types", cuisines)
# ------------------------------------------------------------------------------------------------
    

    online_order = convert_to_binary(online_order)
    book_table = convert_to_binary(book_table)
    
    cost = cost.index(cost_of_tow) + 1
    print("Check if index of cost is Change", cost)
    
    listed_in_type = label_to_listed_type[categories]
    
    input_data = [online_order, book_table, location, rest_type,
                  cost, listed_in_type, loc_city]
    
    cuisines = ' '.join(cuisines)
    
    cuisines_tfidf = tfidf.transform([cuisines]).toarray()
    
    for i in cuisines_tfidf[0]:
        input_data.append(i)
        
        
    #------------------------------prediction part------------
    predict = sl.form_submit_button("predict")


    if predict:
        if cuisines == "" :
            sl.warning("Please Inter yor cuisines type / types")
            print("\n", "-"*100, "\n")
            
        else:
            df = pd.DataFrame(np.array(input_data).reshape(1, 121), columns=column)
    
            X = pipeline.transform(df)
    
            prediction = model.predict(X)[0]
    
            if prediction == 1:
                message = "Coungaturation you will Succeed"
                            
            else:
                message = "Sorry you will fail"
                           
    
            print(message)
            sl.write(message)
            # -------------------------------------------------------------
            
            sl.success("submitted successful")
            
            print("\n", "-"*100, "\n")
        
