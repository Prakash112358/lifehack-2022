from flask import *
import random

app = Flask(__name__)

@app.route('/') #home page
def home():
    return render_template("homepage.html") #Brings to login page

@app.route('/location_disp', methods = ["POST"])
def location_disp(): 
    
    data = request.form #retrieve username & password

    locations = ["Woodlands Ring Road, Block 10, #05-330", "Queensway Road, Block 2, #01-333", "Swee Heng Road, Block 420, #07-090", "Potong Pasir Road, Block 120, #12-530"]
    return render_template("locations.html", location=locations)

@app.route('/locations')
def locations():
	print("HI")
	img = "./static/img.jpg"
	aud = "./static/audio.jpg"
	# Automate a random image with bbox and store in var 'img' + automate a random audio 'aud'
	# Remember to put the random image and audio in the static folder and img and aud var should only contain string of the name of img/aud : ex: horse.mp3
	return render_template("detection.html", img = img, aud = aud)

@app.route('/keylock', methods = ["POST"])
def keylock():

	digit1 = random.randint(1,9)
	digit2 = random.randint(0,9)
	digit3 = random.randint(0,9)
	digit4 = random.randint(0,9)
	digit5 = random.randint(0,9)
	number = str(digit1) + str(digit2) + str(digit3) + str(digit4) + str(digit5) 
	return render_template("keyLock.html", code=number)

if __name__ == '__main__':
	app.run(port = 5000, debug = True)

