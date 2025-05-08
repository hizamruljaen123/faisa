#web
from flask import Flask, render_template, jsonify, request, url_for, redirect, session, send_file
from flask_session import Session

import json
#import library yg diperlukan

#pengolahan data
import pandas as pd

#textprocessing
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
import mysql.connector

#tampilkan data ke web
import json

#klasifikasi
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree

#evaluasi
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#gambar
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from wordcloud import WordCloud

#scrape tokopedia
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup, Tag
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


#buat web
app = Flask(__name__)
app.secret_key = 'any random string'
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

#koneksi ke database mysql
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="bela_tokopedia"
)

cursor = mydb.cursor()

#scrape proses
class ScrapeTokopedia:

    def __init__(self, toko):
        self.toko = "https://www.tokopedia.com/"+toko+"/review"
    
    def scraping_data(self, bintang=None, diambil=3000, rentang_waktu=None):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(self.toko)
        
        try:
            wait = WebDriverWait(driver, 10)
            bintang_5 = wait.until(EC.presence_of_element_located((By.XPATH, '//div[@id="content-Rating"]//div[1]//label[1]')))
            bintang_4 = wait.until(EC.presence_of_element_located((By.XPATH, '//div[@id="content-Rating"]//div[1]//label[2]')))
            bintang_3 = wait.until(EC.presence_of_element_located((By.XPATH, '//div[@id="content-Rating"]//div[1]//label[3]')))
            bintang_2 = wait.until(EC.presence_of_element_located((By.XPATH, '//div[@id="content-Rating"]//div[1]//label[4]')))
            bintang_1 = wait.until(EC.presence_of_element_located((By.XPATH, '//div[@id="content-Rating"]//div[1]//label[5]')))
        except Exception as e:
            print(f"Error finding rating elements: {e}")
            bintang_5 = bintang_4 = bintang_3 = bintang_2 = bintang_1 = None

        if bintang == 1:
            bintang_1.click()
        elif bintang == 2:
            bintang_2.click()
        elif bintang == 3:
            bintang_3.click()
        elif bintang == 4:
            bintang_4.click()
        elif bintang == 5:
            bintang_5.click()
        else:
            None
        
        time.sleep(2)
        #dictionary data
        datas = {
            'nama_toko' : [],
            'rating_toko' : [],
            'user_name' :[],
            'user_review' : [],
            'user_rating' : [],
            'user_product':[],
            'user_date': []
        }

        # Get current date for filtering
        now = pd.Timestamp.now()
        if rentang_waktu == '1_bulan':
            time_ago = now - pd.DateOffset(months=1)
        elif rentang_waktu == '1_tahun':
            time_ago = now - pd.DateOffset(years=1)
        else:
            time_ago = now - pd.DateOffset(years=1)
        # iterasi sesuai jumlah pagination
        for i in range(0, round(diambil / 10)):

            try:
                driver.find_element(By.CSS_SELECTOR, 'button.css-89c2tx').click()
            except:
                None

            
            soup = BeautifulSoup(driver.page_source, "html.parser")
            nama_toko = soup.h1.text
            rating_toko_element = soup.find('span', attrs={'class':'score'})
            rating_toko = rating_toko_element.text if rating_toko_element else 'N/A'
            
            containers = soup.findAll('article', attrs={'class':'css-ccpe8t'})
            #ambil data
            for container in containers:

                try:
                    review = container.find('span', attrs={'data-testid':'lblItemUlasan'}).text
                except:
                    review = container.find('p', attrs={'class':'css-zhjnk4-unf-heading e1qvo2ff8'})

                username = container.find('span', attrs={'class':"name"}).text
                rating = container.find('div', attrs={"data-testid":"icnStarRating"}).attrs['aria-label'].split(' ')[-1]
                date = container.find('p', attrs={"class":"css-1dfgmtm-unf-heading e1qvo2ff8"}).text
                product = container.find('a', attrs={"class": "styProduct"}).text

                # Parse the date string with multiple formats
                date_formats = ['%d %b %Y', '%d %B %Y', '%Y-%m-%d', '%d/%m/%Y']
                date_obj = None
                for fmt in date_formats:
                    try:
                        date_obj = pd.to_datetime(date, format=fmt)
                        break  # If parsing succeeds, break the loop
                    except ValueError:
                        continue  # If parsing fails, try the next format

                if date_obj is None:
                    # Handle cases where the date format is not recognized
                    print(f"Warning: Could not parse date string: {date}")
                    continue  # Skip this review if the date is invalid

                # Filter reviews based on date
                if date_obj >= time_ago:
                    datas['nama_toko'].append(nama_toko)
                    datas['rating_toko'].append(rating_toko)
                    datas['user_name'].append(username)
                    datas['user_review'].append(review)
                    datas['user_rating'].append(rating)
                    datas['user_product'].append(product)
                    datas['user_date'].append(date)

            # klik pagination
            time.sleep(2)
            try:
                next_button = driver.find_element(By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']")
                next_button.click()
                time.sleep(3)
            except Exception as e:
                print(f"Error clicking next button: {e}")
                break  # If next button is not found, break the loop
        
        driver.quit()
        
        return pd.DataFrame(datas)

#logout
@app.route("/keluar")
def keluar():
    session.pop("admin",None)
    session['name'] = False
    return redirect(url_for("index"))

#login
@app.route("/", methods=["GET","POST"])
def index():
    if 'admin' in session:
        return redirect(url_for("dashboard"))
    if request.method == 'POST':
        username = request.form["username"]
        password = request.form["password"]
        if len(username)== 0 or len(password) == 0:
            return render_template("login.html",err="Masukkan data login!")
        else:
            cursor = mydb.cursor()
            cursor.execute("SELECT * FROM auth WHERE username='{}'".format(username))
            myresult = cursor.fetchone()
            if myresult == None:
                return render_template("login.html",err="Tidak ditemukan user tersebut!")
            else:
                if myresult[0]== username and myresult[1] == password:
                    session["admin"] = True
                    session["nama"] = myresult[2]
                    return redirect(url_for("dashboard"))
                else:
                    return render_template("login.html",err="Password yang dimasukkan salah!")
        return "Try Again"
    return render_template("login.html")

#masuk menu home
@app.route('/dashboard')
def dashboard():
    if 'admin' not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html", title='Selamat Datang')

#tampilkan dataset
@app.route('/dataset')
def dataset():
    if 'admin' not in session:
        return redirect(url_for("index"))

    cursor.execute("SELECT * FROM dataset")
    myresult = cursor.fetchall()

    arr = []
    count=0
    for x in myresult:
        count=count+1
        arr.append({"no":count, "id":x[0], "username":x[3], "text":x[4],"product":x[6],"rating":x[5]})
    
    return render_template('dataset.html',data=arr, title='Dataset')


#scrape dataset
@app.route('/scrapedataset', methods=['POST'])
def scrapedataset():
    if 'admin' not in session:
        return redirect(url_for("index"))
    if request.method == 'POST':

        id_toko = str(request.form["id_toko"])
        star = int(request.form['bintang'])
        diambil = int(request.form['diambil'])
        rentang_waktu = request.form['rentang_waktu']

        dataset = ScrapeTokopedia(id_toko).scraping_data(bintang=star, diambil=diambil, rentang_waktu=rentang_waktu)


        if request.form['Scrape'] == "Ambil Data":

            sql = "INSERT INTO dataset (nama_toko, rating_toko, user_name, text, user_rating, user_product, user_date) VALUES (%s,%s,%s,%s,%s,%s,%s)"

            tupp = []
            for idx, row in dataset.iterrows():

                if isinstance(row['user_review'], Tag) ==  True:
                    row['user_review'] = row['user_review'].text

                tupp.append((row['nama_toko'],row['rating_toko'],row['user_name'],row['user_review'], row['user_rating'], row['user_product'],row['user_date']))

            cursor.executemany(sql,tupp)
            mydb.commit()

            return redirect(url_for("dataset"))

        else:
            file_path = f'scrape_file/{id_toko}.xlsx'
            file_to_excel = dataset.to_excel(file_path)
            file_name = id_toko+'.xlsx'
            return send_file(file_path, download_name=file_name)

#import dataset proses
@app.route('/importdataset', methods=['GET','POST'])
def importdataset():
    if 'admin' not in session:
        return redirect(url_for("index"))
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(url_for('dataset'))
        file = request.files['file']
        excel = pd.read_excel(file)

        sql = "INSERT INTO dataset (nama_toko, rating_toko, user_name, text, user_rating, user_product, user_date) VALUES (%s,%s,%s,%s,%s,%s,%s)"

        tupp = []
        c=-1
        for x in excel["user_review"]:

            if isinstance(x, str) == False :
                try:
                    x = x.text
                except:
                    x = ""

            c=c+1
            tupp.append((excel['nama_toko'][c],float(excel['rating_toko'][c]),excel['user_name'][c],x, int(excel['user_rating'][c]), excel['user_product'][c],excel['user_date'][c]))

        
        cursor.executemany(sql,tupp)
        mydb.commit()

        return redirect(url_for("dataset"))

#hapus dataset
@app.route("/hapusdataset", methods=["POST"])
def hapusdataset():
    if 'admin' not in session:
        return redirect(url_for("index"))

    if request.method == 'POST':
        cursor.execute("TRUNCATE TABLE dataset")
        mydb.commit()

        cursor.execute("TRUNCATE TABLE textprocessing")
        mydb.commit()

        return redirect(url_for("dataset"))

#hapus satu data
@app.route("/hapus/<id>", methods=["GET"])
def hapus(id):
    if 'admin' not in session:
        return redirect(url_for("index"))

    if request.method == 'GET':
        cursor = mydb.cursor()
        cursor.execute("DELETE FROM dataset where id="+id)
        mydb.commit()

        return redirect(url_for("dataset"))

#tampilkan hasil proses text
@app.route('/textprocessing')
def textprocessing():
    if 'admin' not in session:
        return redirect(url_for("index"))

    cursor.execute("SELECT * FROM textprocessing")
    myresult = cursor.fetchall()

    arr = []
    count=0
    for x in myresult:
        count=count+1
        arr.append({"no":count,"sebelum":x[1],"text":x[2],"sentimen":x[3]})


    textprocessing_xlsx = pd.DataFrame(arr).to_excel('static/data_created/textprocessing.xlsx', index=False)
    
    return render_template("textprocessing.html",data=arr, title='Text Processing')

#text preprocessing proses
@app.route("/prosestext")
def prosestext():
    if 'admin' not in session:
        return redirect(url_for("index"))
    
    cursor.execute("SELECT * FROM dataset")
    myresult = cursor.fetchall()

    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    steming = StemmerFactory()
    stemmer = steming.create_stemmer()
    
    indo_list = stopwords.words('indonesian')

    alay = pd.read_excel("kamus/kamus-alay.xlsx")

    alay_dict = {}

    for index, row in alay.iterrows():
        if row['slang'] not in alay_dict:
            alay_dict[row['slang']] = row['formal'] 

    payload = []

    for x in myresult:

        # melakukan case folding
        case_folding = x[4].lower()

        # cleaning
        dua = re.sub(r"@[^\s]+"," ",case_folding)
        dua = re.sub(r"#[^\s]+"," ",dua)
        dua = re.sub(r"\."," ",dua)
        dua = re.sub(r"http[^\s]+"," ",dua)
        dua = re.sub(r"\?"," ",dua)
        dua = re.sub(r","," ",dua)
        dua = re.sub(r"”"," ",dua)
        dua = re.sub(r"co/[^\s]+"," ",dua)
        dua = re.sub(r":'\)"," ",dua)
        dua = re.sub(r":\)","",dua)
        dua = re.sub(r"&"," ",dua)
        dua = re.sub(r'\"([^\"]+)\"',"\g<1>",dua)
        dua = re.sub(r'\([^\)]+\"',"",dua)
        dua = re.sub(r'\((.+)\)',"\g<1>",dua)
        dua = re.sub(r'-'," ",dua)
        dua = re.sub(r':\('," ",dua)
        dua = re.sub(r':'," ",dua)
        dua = re.sub(r'\('," ",dua)
        dua = re.sub(r'\)'," ",dua)
        dua = re.sub(r"'"," ",dua)
        dua = re.sub(r'"'," ",dua)
        dua = re.sub(r';'," ",dua)
        dua = re.sub(r':v'," ",dua)
        dua = re.sub(r'²'," ",dua)
        dua = re.sub(r':"\)'," ",dua)
        dua = re.sub(r'\[\]'," ",dua)
        dua = re.sub(r'“',"",dua)
        dua = re.sub(r'_'," ",dua)
        dua = re.sub(r'—'," ",dua)
        dua = re.sub(r'…'," ",dua)
        dua = re.sub(r'='," ",dua)
        dua = re.sub(r'\/'," ",dua)
        dua = re.sub(r'\[\w+\]'," ",dua)
        dua = re.sub(r'!'," ",dua)
        dua = re.sub(r"'"," ",dua)
        dua = re.sub(r'\s+'," ",dua)
        dua = re.sub(r'^RT',"",dua) 
        dua = re.sub(r'\s+$',"",dua)   
        dua = re.sub(r'^\s+',"",dua)  

        #tokenisasi
        tokenisasi = word_tokenize(dua)

        #ganti kata alay / normalisasi
        normalisasi = ' '.join([alay_dict[term] if term in alay_dict else term for term in tokenisasi])

        # filtering dengan sastrawi
        filtering = stopword.remove(normalisasi) 
        #filtering = [kata for kata in filtering if kata not in indo_list]

        #stemming
        stemming = stemmer.stem(filtering)

        #save
        payload.append((stemming,x[4]))

    #kosongkan dataset lama
    cursor.execute("truncate table textprocessing")
    mydb.commit()

    #masukkan dataset baru
    sql = "INSERT INTO textprocessing(text,sebelum, sentimen) VALUES (%s,%s, %s)"
    payload_with_sentiment = [(text, sebelum, 'Positif') for text, sebelum in payload]
    cursor.executemany(sql, payload_with_sentiment)
    mydb.commit()
    return redirect(url_for("textprocessing"))


#proses labelisasi
@app.route('/labelisasi/<id>',  methods=['GET'])
def labelisasi(id):
    if 'admin' not in session:
        return redirect(url_for("index"))

    if id == 'lexicon':

        lexicon_negatif = pd.read_excel('kamus/negative.xlsx')
        lexicon_positif = pd.read_excel('kamus/positive.xlsx')

        lexicon = pd.concat([lexicon_negatif, lexicon_positif], axis=0)

        lexicon_dict = {}

        for index, row in lexicon.iterrows():
                lexicon_dict[row[0]] = int(row[1])

        def sentiment_analysis_lexicon_indonesia(text):
            score = 0
            for word in text:
                if (word in lexicon_dict):
                    score = score + lexicon_dict[word]

            if (score > 0):
                polarity = 'Positif'
            else:
                polarity = 'Negatif'

            return score, polarity

        cursor.execute("SELECT * FROM textprocessing")
        myresult = cursor.fetchall()

        arr = []
        for x in myresult:
            text = word_tokenize(x[2])

            analisa = sentiment_analysis_lexicon_indonesia(text)[1]

            sql = f"UPDATE textprocessing SET sentimen = '{analisa}' WHERE id = '{x[0]}'"
            cursor.execute(sql)
            mydb.commit()
        
        return redirect(url_for("textprocessing"))

    else:

        def rating(rate):
            if rate > 3 and rate <= 5:
                label = 'Positif'
            else:
                label = 'Negatif'

            return label

        cursor.execute("SELECT * FROM dataset")
        myresult = cursor.fetchall()

        arr = []
        for x in myresult:

            analisa = rating(x[5])

            sql = f"UPDATE textprocessing SET sentimen = '{analisa}' WHERE id = '{x[0]}'"
            cursor.execute(sql)
            mydb.commit()
        
        return redirect(url_for("textprocessing"))

#bagi data
@app.route('/split')
def split():
    if 'admin' not in session:
        return redirect(url_for("index"))

    query = "SELECT * FROM textprocessing"

    cursor.execute(query)
    myresult = cursor.fetchall()

    X = []
    y = []

    for l in myresult:
        X.append(l[2])
        y.append(l[3])

    p = False
    q = False
    err = False

    try:
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, train_size=0.7, random_state=0, stratify=y)
        vectorizer = TfidfVectorizer()
        
        c=-1
        p = []
        for x in X_train:
            c=c+1
            p.append({"no":c+1,"text":X_train[c], "sentimen":y_train[c]})

        d=-1
        q = []
        for x in X_test:
            d=d+1
            q.append({"no":d+1,"text":X_test[d], "aktual":y_test[d]})


        datatraining_xlsx = pd.DataFrame(p).to_excel('static/data_created/datatraining.xlsx', index=False)
        datatesting_xlsx = pd.DataFrame(q).to_excel('static/data_created/datatesting.xlsx', index=False)

    except:
        err = "Silahkan lakukan import data dan proses text terlebih dahulu!"
    
    return render_template("split.html", datatraining=p, datatesting=q, title='Split Data', err=err)


# melakukan klasifikasi
@app.route('/klasifikasi')
def klasifikasi():
    if 'admin' not in session:
        return redirect(url_for("index"))
    
    p = False
    cmatrix_dump = False
    err = False
    akurasi = 0  # Initialize akurasi

    cursor.execute("SELECT * FROM textprocessing")
    myresult = cursor.fetchall()

    X = []
    y = []

    for l in myresult:
        X.append(l[2])
        y.append(l[3])

    try:
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, train_size=0.7, random_state=0, stratify=y)
        vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, sublinear_tf=True, use_idf=True)

        X_train_tf = vectorizer.fit_transform(X_train)
        X_test_tf = vectorizer.transform(X_test)

        model = DecisionTreeClassifier(criterion='entropy', random_state=0)
        model.fit(X_train_tf, y_train)
        result = model.predict(X_test_tf)

        c=-1
        p = []
        for x in result:
            c=c+1
            p.append({"no":c+1,"text":X_test[c], "aktual":y_test[c], "prediksi":x})
                
        akurasi = accuracy_score(y_test, result)

        klasifikasi_xlsx = pd.DataFrame(p).to_excel('static/data_created/klasifikasi.xlsx', index=False)

    except:
        err = "Silahkan inputkan data dan lakukan proses text terlebih dahulu!"
    
    return render_template("klasifikasi.html", data=p, akurasi = round(akurasi*100,2), title='Hasil Klasifikasi', err=err)

#tampilkan evaluasi
@app.route('/evaluasi')
def evaluasi():
    if 'admin' not in session:
        return redirect(url_for("index"))

    cursor.execute("SELECT * FROM textprocessing")
    myresult = cursor.fetchall()

    X = []
    y = []

    for l in myresult:
        X.append(l[2])
        y.append(l[3])

    cmatrix = False
    cmatrix_dump = False
    performa = False
    err = False

    try:
        #bagi dataset jadi data training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, train_size=0.7, random_state=0, stratify=y)
        vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, sublinear_tf=True, use_idf=True)

        X_train_tf = vectorizer.fit_transform(X_train)
        X_test_tf = vectorizer.transform(X_test)


        #uji metode
        model = DecisionTreeClassifier(criterion='entropy', random_state=0)
        model.fit(X_train_tf, y_train)
        result = model.predict(X_test_tf)

        c=-1
        p = []
        for x in result:
            c=c+1
            p.append({"no":c+1,"text":X_test[c],"sentimen":x})
        
        matrix = confusion_matrix(y_test,result,labels=["Positif","Negatif"])

    
        cmatrix = [{ "kosong": "Aktual Positif", "actualpositif": int(matrix[0][0]), "actualnegatif": int(matrix[0][1]) },
               { "kosong": "Aktual Negatif", "actualpositif": int(matrix[1][0]), "actualnegatif": int(matrix[1][1]) }
               ]

        akurasi = accuracy_score(y_test, result)

        presisi = precision_score(y_test, result, labels=["Positif","Negatif"], average='macro')

        rikol = recall_score(y_test, result, labels=["Positif","Negatif"], average='macro')

        efwan = f1_score(y_test, result, labels=["Positif","Negatif"], average='macro')

        performa = {"akurasi": round(akurasi*100,2), "presisi": round(presisi*100,2), "rikol": round(rikol*100,2), "efwan" : round(efwan*100,2)}

    except:
        err = 'Silahkan lakukan import data dan proses text terlebih dahulu!'
        cmatrix = []
        performa = {}

    # Generate donut chart 1
    labels1 = ['Positif', 'Negatif']
    sizes1 = [cmatrix[0]['actualpositif'] if cmatrix else 0, cmatrix[1]['actualnegatif'] if cmatrix else 0]
    colors1 = ['#26B99A', '#3498DB']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes1, labels=labels1, colors=colors1, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # Convert plot to PNG image
    pngImage1 = io.BytesIO()
    FigureCanvas(fig1).print_png(pngImage1)
    # Encode PNG image to base64 string
    pngImageB64String1 = "data:image/png;base64,"
    pngImageB64String1 += base64.b64encode(pngImage1.getvalue()).decode('utf8')
    plt.close(fig1)

    # Generate donut chart 2
    labels2 = ['Sesuai', 'Tidak Sesuai']
    sizes2 = [cmatrix[0]['actualpositif'] + cmatrix[1]['actualnegatif'] if cmatrix else 0, (cmatrix[0]['actualnegatif'] + cmatrix[1]['actualpositif']) if cmatrix else 0]
    colors2 = ['#26B99A', '#3498DB']
    fig2, ax2 = plt.subplots()
    ax2.pie(sizes2, labels=labels2, colors=colors2, autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.axis('equal')
    # Convert plot to PNG image
    pngImage2 = io.BytesIO()
    FigureCanvas(fig2).print_png(pngImage2)
    # Encode PNG image to base64 string
    pngImageB64String2 = "data:image/png;base64,"
    pngImageB64String2 += base64.b64encode(pngImage2.getvalue()).decode('utf8')
    plt.close(fig2)

    return render_template("pengujian.html", cmatrix=cmatrix, performa = performa, title='Evaluasi Algoritma', err=err, image1=pngImageB64String1, image2=pngImageB64String2)


#tampilkan sebaran kata
@app.route("/wordcloud")
def wordcloud():

    cursor.execute("SELECT * FROM textprocessing")
    data = cursor.fetchall()

    pngImageB64String = False
    err = False
    sentiment_pct = [0, 0]
    sentiment_pct_json = ""
    kata = []
    label = []

    for l in data:
        kata.append(word_tokenize(str(l[2])))
        label.append(l[3])

    tweets = [word for tweet in kata for word in tweet]

    kata_kata = " ".join(tweets)

    WC = WordCloud(max_words=100, background_color="white", max_font_size = 80)
    

    try:

        pos = len([x for x in label if x == 'Positif'])
        neg = len([x for x in label if x == 'Negatif'])

        WC.generate(kata_kata)
        # Generate plot
        fig = Figure(figsize=(8,4))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(WC)
        ax.axis("off")
        
        # Convert plot to PNG image
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        
        # Encode PNG image to base64 string
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')


    except:
        err = 'Silahkan lakukan import data dan proses text terlebih dahulu'
        sentiment_pct_json = json.dumps([0, 0])
        return render_template("wordcloud.html", image=pngImageB64String, title='Sebaran Kata', err=err, sentiment_pct=sentiment_pct_json)
    
    pos = len([x for x in label if x == 'Positif'])
    neg = len([x for x in label if x == 'Negatif'])
    sentiment_pct = [pos, neg]
    sentiment_pct_json = json.dumps(sentiment_pct)
    return render_template("wordcloud.html", image=pngImageB64String, title='Sebaran Kata', err=err, sentiment_pct=sentiment_pct_json)
# tree plot
@app.route("/tree_plot")
def tree_plot():

    if 'admin' not in session:
        return redirect(url_for("index"))

    err = False

    cursor.execute("SELECT * FROM textprocessing")
    myresult = cursor.fetchall()

    X = []
    y = []

    for l in myresult:
        X.append(l[2])
        y.append(l[3])

    try:
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, train_size=0.7, random_state=0, stratify=y)
        vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, sublinear_tf=True, use_idf=True)

        X_train_tf = vectorizer.fit_transform(X_train)
        X_test_tf = vectorizer.transform(X_test)

        vars = vectorizer.get_feature_names_out()

        model = DecisionTreeClassifier(criterion='entropy', random_state=0)
        model.fit(X_train_tf, y_train)
        result = model.predict(X_test_tf)

        # Generate plot
        plt.figure(figsize=(50,25))
        plot_tree(model,
                  feature_names=list(vars),
                  class_names=['Positif','Negatif'],
                  rounded=True, # Rounded node edges
                  filled=True, # Adds color accoding to class
                  proportion=True);
        plt.savefig('static/plot_tree.svg', dpi=300)
    
    except:
        err = "Silahkan inputkan data dan lakukan proses text terlebih dahulu!"
    
    
    return render_template("tree_plot.html", title='Tree Plot', err=err)
        

if __name__=='__main__':
    app.run(debug=True)