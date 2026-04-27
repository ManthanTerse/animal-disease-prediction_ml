from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import pandas as pd
import joblib
import csv
import os
from datetime import datetime
from preprocess_utils import preprocess_df
from feature_engineering import add_engineered_features

app = Flask(__name__)
app.secret_key = "manthan"

CONTACT_FILE = "data/contact_messages.csv"

def save_message(name, email, message):
    file_exists = os.path.exists(CONTACT_FILE)
    with open(CONTACT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "name", "email", "message"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, email, message
        ])

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        if not name or not email or not message:
            flash("Please fill all fields")
            return redirect(url_for("contact"))

        save_message(name, email, message)
        flash("Message saved successfully")
        return redirect(url_for("contact"))

    return render_template("contact.html")

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

MODEL_PATH = "model.pkl"
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
feature_columns = model_data["feature_columns"]

def format_text(value):
    return (value or "").strip().title()

@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    data = {
        "Animal_Type": format_text(form.get("animal_type")),
        "Breed": format_text(form.get("breed")),
        "Age": form.get("age"),
        "Gender": format_text(form.get("gender")),
        "Weight": form.get("weight"),
        "Symptom_1": format_text(form.get("symptom1")),
        "Symptom_2": format_text(form.get("symptom2")),
        "Symptom_3": format_text(form.get("symptom3")),
        "Symptom_4": format_text(form.get("symptom4")),
        "Duration": form.get("duration"),
        "Appetite_Loss": form.get("appetite_loss") or "no",
        "Vomiting": form.get("vomitting") or "no",
        "Diarrhea": form.get("diarrhea") or "no",
        "Coughing": form.get("coughing") or "no",
        "Labored_Breathing": form.get("laboured_breathing") or "no",
        "Lameness": form.get("lameness") or "no",
        "Skin_Lesions": form.get("skin_lesions") or "no",
        "Nasal_Discharge": form.get("nasal_discharge") or "no",
        "Eye_Discharge": form.get("eye_discharge") or "no",
        "Body_Temperature": form.get("body_temperature"),
        "Heart_Rate": form.get("heart_rate"),
    }

    input_df = pd.DataFrame([data])
    engineered_input = add_engineered_features(preprocess_df(input_df))
    input_df = input_df.reindex(columns=feature_columns)
    prediction = model.predict(input_df)[0]

    result_details = {
        "Severity_Level": engineered_input.at[0, "Severity_Level"],
        "Temperature_Category": engineered_input.at[0, "Temperature_Category"],
        "Heart_Rate_Category": engineered_input.at[0, "Heart_Rate_Category"],
    }

    return render_template(
        "prediction.html",
        prediction=prediction,
        result_details=result_details,
    )

# ======================= CHATBOT =======================
@app.route("/guide_chatbot", methods=["POST"])
def guide_chatbot():
    data = request.json
    msg = data.get("message", "").lower().strip()

    # ---- GREETINGS ----
    if any(w in msg for w in ["hi", "hello", "hey", "hiya", "howdy", "yo", "sup", "what's up", "whats up",
                               "good morning", "good afternoon", "good evening", "greetings", "helo", "hii",
                               "hiii", "hai", "hello there", "hi there"]):
        return jsonify({"reply": "Hey there! 👋 Welcome to the Pet Disease Prediction Portal! I'm your pet health assistant. Ask me anything about your pet's health, symptoms, or how to use this site 🐾"})

    # ---- HOW ARE YOU ----
    if any(w in msg for w in ["how are you", "how r you", "how are u", "you ok", "you good", "hows it going",
                               "how's it going", "doing well", "how do you do", "you alright", "all good",
                               "how you doing", "how r u"]):
        return jsonify({"reply": "I'm doing great, thanks for asking! 😄 Ready to help you and your furry friend. What's on your mind?"})

    # ---- WHO ARE YOU ----
    if any(w in msg for w in ["who are you", "what are you", "are you a bot", "are you human", "are you real",
                               "are you ai", "are you robot", "what is your name", "your name",
                               "introduce yourself", "tell me about yourself", "who made you", "who built you",
                               "who created you"]):
        return jsonify({"reply": "I'm Pet Assistant 🤖 — an AI chatbot built to help you navigate this pet disease prediction portal and answer your pet health questions! I was built as part of the Pet Disease Prediction Project."})

    # ---- THANKS ----
    if any(w in msg for w in ["thanks", "thank you", "thank u", "thx", "ty", "tysm", "appreciate", "cheers",
                               "gracias", "merci", "shukriya", "thnx", "thankful", "grateful", "many thanks"]):
        return jsonify({"reply": "You're welcome! 😊 Always happy to help. Feel free to ask anything else anytime!"})

    # ---- BYE ----
    if any(w in msg for w in ["bye", "goodbye", "see you", "see ya", "cya", "take care", "later", "good night",
                               "goodnight", "tata", "ttyl", "have a good day", "have a nice day",
                               "gotta go", "got to go", "leaving now"]):
        return jsonify({"reply": "Goodbye! 👋 Take good care of your pet. Come back anytime you need help! 🐾"})

    # ---- ABOUT THE SITE ----
    if any(w in msg for w in ["what is this", "about this site", "about this portal", "what does this do",
                               "what is this website", "purpose", "what is this app", "tell me about",
                               "about the project", "this project", "what does the site do", "portal info",
                               "what can this do"]):
        return jsonify({"reply": "This is the Pet Disease Prediction Portal 🐾 — an AI-powered tool that helps you identify possible diseases in your pet based on symptoms. Go to the Prediction page, enter your pet's details, and get instant results!"})

    # ---- HOW TO USE ----
    if any(w in msg for w in ["how to use", "how do i use", "how does it work", "how to start",
                               "where do i start", "instructions", "guide me", "help me use", "steps",
                               "what to do", "how to predict", "how to check", "walk me through",
                               "explain how", "show me how"]):
        return jsonify({"reply": "Here's how to use it 📋: 1️⃣ Go to the Disease Prediction page. 2️⃣ Enter your pet's details like type, breed, age, and weight. 3️⃣ Fill in the symptoms your pet is showing. 4️⃣ Hit submit and get the prediction instantly! 🎯"})

    # ---- PREDICTION PAGE ----
    if any(w in msg for w in ["prediction page", "where to predict", "find prediction", "go to prediction",
                               "prediction form", "disease form", "predict page", "where is prediction"]):
        return jsonify({"reply": "You can find the Disease Prediction page in the top navigation bar! Just click on 'Disease Prediction' and fill in your pet's info 🐶🐱"})

    # ---- DOG QUESTIONS ----
    if any(w in msg for w in ["my dog", "dog is sick", "dog not eating", "dog vomiting", "dog coughing",
                               "dog diarrhea", "dog fever", "dog tired", "dog lazy", "dog losing hair",
                               "dog scratching", "dog limping", "dog not drinking", "dog bleeding",
                               "dog eyes", "dog nose", "puppy sick", "puppy not eating", "dog is unwell",
                               "dog seems off", "dog acting weird", "dog in pain", "dog not moving",
                               "dog very quiet"]):
        return jsonify({"reply": "Oh no, sorry to hear your dog isn't feeling well 🐶💙 Please head to the Disease Prediction page and enter your dog's symptoms. Our AI will help identify what might be wrong. If symptoms are severe, please visit a vet immediately!"})

    # ---- CAT QUESTIONS ----
    if any(w in msg for w in ["my cat", "cat is sick", "cat not eating", "cat vomiting", "cat coughing",
                               "cat diarrhea", "cat fever", "cat tired", "cat lazy", "cat losing hair",
                               "cat scratching", "cat limping", "cat not drinking", "cat bleeding",
                               "cat eyes", "cat sneezing", "kitten sick", "kitten not eating",
                               "cat acting weird", "cat in pain", "cat not moving", "cat very quiet",
                               "cat seems off", "cat unwell"]):
        return jsonify({"reply": "Aww, hope your cat feels better soon 🐱💛 Please use the Disease Prediction page and enter your cat's symptoms. Our AI will give you possible disease insights. For urgent issues, consult a vet right away!"})

    # ---- RABBIT ----
    if any(w in msg for w in ["rabbit", "bunny", "my rabbit", "rabbit sick", "bunny sick"]):
        return jsonify({"reply": "We currently support dogs and cats for disease prediction 🐶🐱 For rabbits and other pets, we recommend visiting a qualified veterinarian directly 🐰"})

    # ---- OTHER ANIMALS ----
    if any(w in msg for w in ["bird", "parrot", "fish", "hamster", "guinea pig", "turtle", "snake",
                               "lizard", "horse", "cow", "goat", "sheep", "pig", "tortoise",
                               "ferret", "hedgehog"]):
        return jsonify({"reply": "Currently our prediction model supports dogs and cats only 🐶🐱 For other animals, please consult a veterinarian. We hope to expand support in the future!"})

    # ---- WHICH ANIMALS SUPPORTED ----
    if any(w in msg for w in ["which animals", "animals supported", "what pets", "supported pets",
                               "which pets", "what animals", "pet types", "animals available"]):
        return jsonify({"reply": "Currently the portal supports 🐶 Dogs and 🐱 Cats for disease prediction. More animals may be added in future updates!"})

    # ---- SYMPTOMS GENERAL ----
    if any(w in msg for w in ["what symptoms", "which symptoms", "common symptoms", "symptom list",
                               "what to enter", "symptom tips", "symptom advice", "how many symptoms",
                               "list of symptoms", "tell me symptoms", "possible symptoms"]):
        return jsonify({"reply": "Common symptoms you can enter include: 🤒 Fever, 🤮 Vomiting, 💩 Diarrhea, 😮‍💨 Coughing, 😴 Lethargy, 🍽️ Loss of Appetite, 👁️ Eye Discharge, 👃 Nasal Discharge, 🩹 Skin Lesions, and 🦵 Limping. Enter as many as apply!"})

    # ---- FEVER ----
    if any(w in msg for w in ["fever", "high temperature", "body temp", "temperature high", "pet is hot",
                               "feels warm", "hot to touch", "overheating", "temp is high",
                               "running a fever"]):
        return jsonify({"reply": "Fever in pets can be serious 🌡️ Normal temperature for dogs and cats is 38–39.2°C. If your pet has a fever, use the Prediction page and also consider visiting a vet, especially if the fever is above 40°C!"})

    # ---- VOMITING ----
    if any(w in msg for w in ["vomiting", "throwing up", "vomit", "puking", "puke", "nausea",
                               "threw up", "keeps vomiting", "vomited", "retching", "dry heaving"]):
        return jsonify({"reply": "Vomiting in pets can have many causes — from eating something bad to more serious conditions 🤢 Enter this as a symptom on the Prediction page. If vomiting is frequent or has blood in it, please see a vet immediately!"})

    # ---- DIARRHEA ----
    if any(w in msg for w in ["diarrhea", "loose stool", "watery stool", "runny poop", "loose motion",
                               "stomach upset", "tummy trouble", "runny stool", "liquid poop",
                               "frequent stool", "soft stool", "poop problem"]):
        return jsonify({"reply": "Diarrhea in pets can be caused by infections, diet changes, or parasites 💩 Keep your pet hydrated. Enter this in the Prediction form and visit a vet if it lasts more than a day or has blood!"})

    # ---- COUGHING ----
    if any(w in msg for w in ["coughing", "cough", "choking", "gagging", "wheezing", "sneezing",
                               "sneezing a lot", "continuous cough", "persistent cough",
                               "keeps coughing", "hacking cough"]):
        return jsonify({"reply": "Persistent coughing or sneezing in pets could indicate respiratory issues or infections 😮‍💨 Please enter this symptom on the Prediction page. If your pet is struggling to breathe, visit a vet right away!"})

    # ---- LETHARGY ----
    if any(w in msg for w in ["tired", "lazy", "lethargic", "lethargy", "not active", "sleeping too much",
                               "no energy", "dull", "weak", "weakness", "not playful", "less active",
                               "lying around", "not moving much", "sluggish", "inactive", "low energy"]):
        return jsonify({"reply": "Lethargy or unusual tiredness in pets can be a sign of illness 😴 Combined with other symptoms, it could point to something needing attention. Use the Prediction page and monitor your pet closely!"})

    # ---- NOT EATING ----
    if any(w in msg for w in ["not eating", "not hungry", "appetite loss", "loss of appetite",
                               "refusing food", "won't eat", "wont eat", "stopped eating", "no appetite",
                               "skipping meals", "not interested in food", "leaving food",
                               "ignoring food", "not touching food"]):
        return jsonify({"reply": "Loss of appetite in pets is a common warning sign 🍽️ It can be linked to many diseases. Enter this on the Prediction page. If your pet hasn't eaten in over 24 hours, consult a vet!"})

    # ---- NOT DRINKING ----
    if any(w in msg for w in ["not drinking", "not drinking water", "no water", "dehydrated",
                               "dehydration", "drinking too much", "excessive thirst",
                               "drinking more than usual", "refusing water", "won't drink"]):
        return jsonify({"reply": "Hydration is very important for pets 💧 Not drinking can indicate illness, while drinking too much can also be a symptom of certain conditions. Check with a vet and use the Prediction page!"})

    # ---- SKIN / HAIR ----
    if any(w in msg for w in ["skin", "rash", "hair loss", "losing fur", "itching", "scratching",
                               "skin lesion", "bald patch", "fleas", "ticks", "worms", "parasite",
                               "mange", "dry skin", "scaly skin", "skin infection", "fur loss",
                               "coat problem"]):
        return jsonify({"reply": "Skin and coat issues like rashes, itching, or hair loss can be caused by allergies, parasites, or infections 🐾 Enter 'Skin Lesions' as a symptom on the Prediction page and consider a vet visit for proper diagnosis!"})

    # ---- EYES ----
    if any(w in msg for w in ["eye", "eyes", "eye discharge", "watery eyes", "red eyes",
                               "eye infection", "cloudy eyes", "swollen eye", "eye problem",
                               "crusty eyes", "eye mucus", "pawing at eyes", "rubbing eyes",
                               "eye irritation"]):
        return jsonify({"reply": "Eye discharge or redness in pets can indicate an infection or allergy 👁️ Enter 'Eye Discharge' as a symptom in the Prediction form. If the eye looks very swollen or your pet is pawing at it, see a vet soon!"})

    # ---- NOSE ----
    if any(w in msg for w in ["nose", "nasal", "runny nose", "nasal discharge", "nose bleeding",
                               "dry nose", "wet nose", "nose problem", "crusty nose",
                               "green nose discharge", "yellow nose discharge", "nosebleed"]):
        return jsonify({"reply": "A runny or discharging nose can be a sign of respiratory infection in pets 👃 Enter 'Nasal Discharge' on the Prediction page. A healthy pet usually has a moist nose — very dry or crusty noses can also indicate illness!"})

    # ---- LIMPING ----
    if any(w in msg for w in ["limping", "lame", "lameness", "not walking", "hurt leg", "injured leg",
                               "leg pain", "paw pain", "swollen paw", "broken leg", "favouring leg",
                               "dragging leg", "not putting weight", "leg injury", "paw injury",
                               "joint pain"]):
        return jsonify({"reply": "Limping or lameness in pets could be due to injury, joint issues, or infections 🦵 Enter 'Lameness' as a symptom. If your pet can't put weight on a leg at all, please visit a vet as soon as possible!"})

    # ---- BREATHING ----
    if any(w in msg for w in ["breathing", "cant breathe", "difficulty breathing", "labored breathing",
                               "heavy breathing", "shortness of breath", "gasping", "breathless",
                               "rapid breathing", "shallow breathing", "open mouth breathing",
                               "panting a lot"]):
        return jsonify({"reply": "Labored or difficulty breathing is a serious emergency sign ⚠️ Please take your pet to a vet immediately if they are struggling to breathe. You can also enter this symptom on the Prediction page."})

    # ---- EMERGENCY ----
    if any(w in msg for w in ["emergency", "urgent", "critical", "dying", "not breathing", "collapsed",
                               "unconscious", "seizure", "fitting", "convulsion", "blood", "bleeding",
                               "accident", "hit by car", "swallowed something", "ate something",
                               "poisoned", "toxic", "ate poison", "snake bite", "not responding",
                               "unresponsive"]):
        return jsonify({"reply": "⚠️ This sounds like an emergency! Please take your pet to the nearest veterinary clinic IMMEDIATELY. Do not wait — quick action can save your pet's life! 🏥 Call your vet on the way."})

    # ---- VET ADVICE ----
    if any(w in msg for w in ["vet", "doctor", "veterinarian", "should i go to vet", "need a vet",
                               "visit doctor", "see a doctor", "animal hospital", "clinic",
                               "animal clinic", "find a vet", "vet near me", "veterinary",
                               "animal doctor"]):
        return jsonify({"reply": "It's always a good idea to consult a veterinarian for a proper diagnosis 🏥 Our tool gives you initial insights, but a vet's physical examination is irreplaceable. Please don't delay if symptoms are severe!"})

    # ---- BREED ----
    if any(w in msg for w in ["breed", "what breed", "my breed", "dog breed", "cat breed",
                               "which breed", "breed matters", "mixed breed", "labrador", "poodle",
                               "persian", "golden retriever", "german shepherd", "bulldog", "beagle",
                               "husky", "indie", "stray", "dachshund", "rottweiler", "pomeranian",
                               "siamese", "maine coon", "british shorthair"]):
        return jsonify({"reply": "Breed can influence the likelihood of certain diseases 🐕 Please enter your pet's breed accurately on the Prediction page for the most relevant results. Mixed breeds are supported too!"})

    # ---- AGE ----
    if any(w in msg for w in ["age", "how old", "puppy", "kitten", "old dog", "old cat", "senior pet",
                               "young pet", "baby pet", "months old", "years old", "pet age",
                               "adult dog", "adult cat"]):
        return jsonify({"reply": "Age is an important factor in disease prediction 🐾 Puppies and kittens are more vulnerable to certain infections, while older pets are prone to chronic conditions. Enter your pet's age accurately on the form!"})

    # ---- WEIGHT ----
    if any(w in msg for w in ["weight", "overweight", "underweight", "obese", "fat pet", "thin pet",
                               "weight loss", "losing weight", "gaining weight", "pet weight",
                               "body weight", "weight issue", "too fat", "too thin"]):
        return jsonify({"reply": "Weight is a key health indicator for pets ⚖️ Sudden weight loss or gain can be a symptom of underlying conditions. Enter your pet's current weight on the Prediction page for a more accurate result!"})

    # ---- PREDICTION RESULT ----
    if any(w in msg for w in ["result", "prediction result", "what does it mean", "understand result",
                               "result meaning", "output", "what is the output", "got a result",
                               "my result", "result says", "prediction says", "what is predicted"]):
        return jsonify({"reply": "The prediction result gives you the most likely disease based on your pet's symptoms 🎯 It's meant for early awareness — always consult a vet to confirm the diagnosis and get proper treatment!"})

    # ---- ACCURACY ----
    if any(w in msg for w in ["accurate", "accuracy", "how accurate", "correct", "reliable", "trust",
                               "trustworthy", "how reliable", "is it correct", "can i trust",
                               "how good is", "model accuracy", "prediction accuracy"]):
        return jsonify({"reply": "Our AI model is trained on a large dataset of pet health records and gives reasonably accurate predictions 📊 However, it's a screening tool — not a replacement for professional veterinary diagnosis. Always consult a vet for confirmation!"})

    # ---- MULTIPLE PETS ----
    if any(w in msg for w in ["multiple pets", "two pets", "more than one pet", "several pets",
                               "predict for two", "two dogs", "two cats", "both pets",
                               "predict for both"]):
        return jsonify({"reply": "You can predict for multiple pets — just submit the form separately for each pet one at a time 🐶🐱 Each prediction is independent!"})

    # ---- DATA PRIVACY ----
    if any(w in msg for w in ["data", "privacy", "safe", "secure", "my info", "personal info",
                               "stored", "confidential", "is it safe", "data stored", "information",
                               "who sees my data", "data shared", "data policy", "is my data safe"]):
        return jsonify({"reply": "Your data is completely safe 🔒 Symptom inputs are only used for generating predictions and are not shared with third parties. Contact form messages are stored securely for follow-up purposes only."})

    # ---- CONTACT PAGE ----
    if any(w in msg for w in ["contact", "reach you", "reach out", "contact page", "send message",
                               "report issue", "complaint", "feedback", "email", "support team",
                               "get in touch", "talk to someone", "message you", "write to you"]):
        return jsonify({"reply": "You can reach us via the Contact page in the navigation bar 📬 Fill in your name, email, and message and we'll get back to you as soon as possible!"})

    # ---- PREDICTION SPEED ----
    if any(w in msg for w in ["how long", "how fast", "prediction time", "speed", "instant", "quick",
                               "fast result", "waiting", "takes time", "how quick", "response time"]):
        return jsonify({"reply": "Predictions are almost instant ⚡ Once you submit the symptom form, the AI processes it in milliseconds and shows you the result right away!"})

    # ---- TREATMENT ----
    if any(w in msg for w in ["treatment", "medicine", "medication", "cure", "how to treat",
                               "what medicine", "remedy", "give medicine", "tablet", "injection",
                               "vaccine", "antibiotic", "what drug", "what to give", "how to cure",
                               "home remedy"]):
        return jsonify({"reply": "We provide disease predictions, not treatment recommendations 🏥 For treatment, please consult a licensed veterinarian. Never give your pet human medicines without vet approval — it can be dangerous!"})

    # ---- PREVENTION ----
    if any(w in msg for w in ["prevent", "prevention", "how to prevent", "keep healthy", "avoid disease",
                               "pet care tips", "health tips", "how to keep pet healthy", "healthy pet",
                               "preventive care", "routine care"]):
        return jsonify({"reply": "Great question! 💡 To keep your pet healthy: ✅ Regular vet checkups, ✅ Keep vaccinations up to date, ✅ Deworm regularly, ✅ Feed a balanced diet, ✅ Keep them clean and active, ✅ Watch for early symptoms!"})

    # ---- FOOD / DIET ----
    if any(w in msg for w in ["food", "diet", "what to feed", "feeding", "can i feed", "good food",
                               "bad food", "nutrition", "meal", "dog food", "cat food", "raw food",
                               "dry food", "wet food", "treats", "snacks", "what should i feed",
                               "best food for pet"]):
        return jsonify({"reply": "Diet plays a huge role in pet health 🍖 Feed your pet age-appropriate, vet-approved food. Avoid onions, garlic, chocolate, grapes, and raisins — these are toxic! Consult a vet for a proper diet plan."})

    # ---- TOXIC FOODS ----
    if any(w in msg for w in ["chocolate", "grapes", "onion", "garlic", "raisins", "toxic food",
                               "dangerous food", "what not to feed", "foods to avoid",
                               "poisonous food", "harmful food", "can dogs eat", "can cats eat",
                               "xylitol", "avocado", "caffeine", "alcohol"]):
        return jsonify({"reply": "🚫 Foods toxic to pets include: Chocolate, grapes, raisins, onions, garlic, xylitol, avocado, caffeine, and alcohol. Keep these away from your pets always! Call a vet immediately if your pet eats any of these."})

    # ---- WATER ----
    if any(w in msg for w in ["water", "hydration", "how much water", "pet drinking water",
                               "enough water", "water intake", "keep hydrated", "water bowl",
                               "fresh water"]):
        return jsonify({"reply": "Always keep fresh water available for your pet 💧 Dogs generally need about 50ml of water per kg of body weight per day. Cats often need encouragement to drink — try a pet water fountain!"})

    # ---- EXERCISE ----
    if any(w in msg for w in ["exercise", "walk", "activity", "play", "run", "active", "outdoor",
                               "physical activity", "how much exercise", "daily walk", "playtime",
                               "keep active", "exercise routine"]):
        return jsonify({"reply": "Regular exercise is essential for pet health 🏃 Dogs need daily walks and playtime, while cats benefit from interactive toys. Exercise helps maintain healthy weight and mental wellbeing!"})

    # ---- GROOMING ----
    if any(w in msg for w in ["groom", "grooming", "bath", "bathing", "brush", "brushing", "nail",
                               "nail trim", "nail cutting", "clean pet", "wash pet", "hygiene",
                               "cleaning pet", "how often to bathe", "pet hygiene"]):
        return jsonify({"reply": "Regular grooming keeps your pet healthy and comfortable 🛁 Dogs should be bathed every 4–6 weeks. Brush their coat regularly to prevent tangles. Trim nails monthly and clean ears gently. Cats are generally self-grooming but may need help with mats!"})

    # ---- SLEEP ----
    if any(w in msg for w in ["sleep", "sleeping", "how much sleep", "pet sleeping a lot", "sleepy pet",
                               "napping", "rest", "resting too much", "always sleeping"]):
        return jsonify({"reply": "Pets sleep a lot! 😴 Cats sleep 12–16 hours a day, and dogs sleep 12–14 hours. If your pet seems unusually lethargic or can't be woken easily, that could be a health concern — use the Prediction page or see a vet!"})

    # ---- WEATHER ----
    if any(w in msg for w in ["cold", "hot weather", "summer", "winter", "heatstroke", "heat stroke",
                               "overheating", "hypothermia", "too cold", "too hot",
                               "weather effect on pet"]):
        return jsonify({"reply": "Extreme temperatures affect pets significantly 🌡️ In summer, watch for heatstroke — never leave pets in cars! In winter, keep them warm and dry. Short-haired breeds need extra protection in cold weather."})

    # ---- STRESS / ANXIETY ----
    if any(w in msg for w in ["stress", "stressed", "anxiety", "anxious", "scared", "fear", "nervous",
                               "phobia", "thunder", "fireworks", "separation anxiety", "shaking",
                               "trembling", "hiding", "pet hiding", "crying", "whining", "howling"]):
        return jsonify({"reply": "Pets can experience stress and anxiety just like humans 💙 Common triggers include loud noises, changes in routine, or separation. Try calming music, a safe space, or speak to a vet about anxiety management options."})

    # ---- SMELL ----
    if any(w in msg for w in ["smell", "smells bad", "bad odour", "odor", "stinky", "bad breath",
                               "mouth smell", "body odour", "anal glands", "fishy smell",
                               "weird smell"]):
        return jsonify({"reply": "Unusual smells from pets can signal health issues 👃 Bad breath may indicate dental disease, a fishy smell could be anal gland issues, and strong body odour might suggest a skin infection. Mention this to your vet!"})

    # ---- DENTAL ----
    if any(w in msg for w in ["teeth", "tooth", "dental", "gum", "mouth", "oral", "brush teeth",
                               "dental hygiene", "tartar", "plaque", "tooth decay", "broken tooth",
                               "loose tooth"]):
        return jsonify({"reply": "Dental health is very important for pets 🦷 Brush your pet's teeth regularly with pet-safe toothpaste. Bad teeth can lead to serious infections. Annual dental checkups at the vet are highly recommended!"})

    # ---- PREGNANCY / BREEDING ----
    if any(w in msg for w in ["pregnant", "pregnancy", "breeding", "heat", "in heat", "mating",
                               "litter", "puppies", "kittens", "giving birth", "whelping",
                               "spay", "neuter", "sterilise", "sterilize"]):
        return jsonify({"reply": "Pregnancy and breeding in pets requires careful attention 🐣 If your pet is pregnant, ensure regular vet visits. Spaying/neutering is recommended if you're not breeding — it prevents diseases and reduces the stray population!"})

    # ---- PARASITES ----
    if any(w in msg for w in ["fleas", "flea", "tick", "roundworm", "tapeworm", "heartworm",
                               "intestinal worm", "mites", "lice", "infestation", "flea treatment",
                               "deworming", "anti-parasite", "worm treatment"]):
        return jsonify({"reply": "Parasites like fleas, ticks, and worms are common in pets 🪱 Use vet-recommended flea/tick prevention products regularly. Deworm your pet every 3–6 months. Check your pet's coat after outdoor walks for ticks!"})

    # ---- VACCINATION ----
    if any(w in msg for w in ["vaccine", "vaccination", "shots", "immunization", "rabies",
                               "distemper", "parvovirus", "booster", "annual shot",
                               "vaccine schedule", "puppy shots", "kitten shots",
                               "is my pet vaccinated"]):
        return jsonify({"reply": "Vaccinations are crucial for your pet's health 💉 Core vaccines for dogs include Rabies, Distemper, and Parvovirus. For cats — Rabies, Feline Herpesvirus, and Calicivirus. Consult your vet for a proper vaccination schedule!"})

    # ---- INSURANCE ----
    if any(w in msg for w in ["insurance", "pet insurance", "health insurance", "cost",
                               "expensive vet", "vet bill", "afford vet", "vet cost",
                               "how much does vet cost"]):
        return jsonify({"reply": "Pet insurance can help cover unexpected vet bills 💰 Vet costs can be expensive, especially for emergencies. Consider getting pet health insurance early — it's more affordable when your pet is young and healthy!"})

    # ---- MICROCHIP ----
    if any(w in msg for w in ["microchip", "chip", "lost pet", "missing pet", "pet id", "id tag",
                               "lost dog", "lost cat", "find my pet", "pet tracking"]):
        return jsonify({"reply": "Microchipping is one of the best ways to ensure your pet can be identified if lost 🔍 It's a quick, painless procedure done by a vet. Always keep your contact details updated in the microchip registry!"})

    # ---- NEW PET ----
    if any(w in msg for w in ["new pet", "just got a pet", "adopted a pet", "first pet",
                               "first time owner", "new puppy", "new kitten", "new dog", "new cat",
                               "got a puppy", "got a kitten", "tips for new pet", "new pet owner"]):
        return jsonify({"reply": "Congratulations on your new pet! 🎉 First steps: ✅ Schedule a vet checkup within the first week, ✅ Start vaccinations, ✅ Get them microchipped, ✅ Set up a feeding routine, ✅ Give them time to settle in. Enjoy the journey! 🐾"})

    # ---- TRAINING ----
    if any(w in msg for w in ["train", "training", "teach", "behavior", "behaviour", "potty train",
                               "house train", "obedience", "sit", "stay", "bad behavior",
                               "biting", "barking", "aggression", "aggressive pet"]):
        return jsonify({"reply": "Training is key to a happy pet relationship! 🎓 Use positive reinforcement — reward good behaviour with treats and praise. Be consistent and patient. For aggression issues, consider consulting a professional animal behaviourist."})

    # ---- BORED / CASUAL ----
    if any(w in msg for w in ["bored", "nothing to do", "just chatting", "just talking", "wanna chat",
                               "entertain me", "tell me something", "say something", "i am bored",
                               "killing time"]):
        return jsonify({"reply": "Haha I'm a pet health bot, not a comedian 😄 But did you know — a dog's nose print is as unique as a human fingerprint! 🐶👃 Now go spend some quality time with your pet — they'll love it!"})

    # ---- FUN FACTS ----
    if any(w in msg for w in ["fun fact", "fact", "did you know", "interesting", "cool fact",
                               "tell me a fact", "random fact", "pet fact", "animal fact"]):
        return jsonify({"reply": "Here's a fun pet fact! 🐾 Cats have 32 muscles in each ear, allowing them to rotate their ears 180 degrees. And dogs can smell your emotions — they literally sense fear and happiness through scent! 🐱🐶"})

    # ---- COMPLIMENT BOT ----
    if any(w in msg for w in ["you're great", "you are great", "you're awesome", "you are awesome",
                               "good bot", "nice bot", "you're helpful", "well done", "great job",
                               "love you bot", "you're amazing", "best bot", "cool bot"]):
        return jsonify({"reply": "Thank you so much! 😊🙏 I try my best to help you and your furry friends. You're pretty awesome yourself for caring so much about your pet! 🐾❤️"})

    # ---- INSULT BOT ----
    if any(w in msg for w in ["you're bad", "you are bad", "useless", "bad bot", "stupid bot",
                               "dumb", "worst", "you suck", "not helpful", "terrible", "awful",
                               "rubbish", "trash", "pathetic", "hate you"]):
        return jsonify({"reply": "Sorry I didn't help well enough! 😅 Try asking about symptoms, diseases, or how to use the site. If you have feedback, please use the Contact page 🙏"})

    # ---- LOVE / CUTE ----
    if any(w in msg for w in ["cute", "adorable", "love my pet", "i love my dog", "i love my cat",
                               "my pet is cute", "love animals", "pets are the best",
                               "best pet ever", "love him", "love her", "my baby", "fur baby"]):
        return jsonify({"reply": "Aww that's so sweet! 🥰 Pets bring so much joy into our lives. Being a caring owner like you makes all the difference to their health and happiness! 🐾❤️"})

    # ---- SAD / WORRIED ----
    if any(w in msg for w in ["worried", "i'm scared", "i'm worried", "i don't know what to do",
                               "panicking", "freaking out", "what do i do", "please help",
                               "help me", "i need help", "i am worried", "i am scared"]):
        return jsonify({"reply": "It's completely understandable to feel worried about your pet 💙 You're doing the right thing by seeking help! Use our Prediction page for initial insights, and don't hesitate to visit a vet. Your pet is lucky to have such a caring owner!"})

    # ---- PET DIED ----
    if any(w in msg for w in ["my pet died", "pet passed away", "lost my pet", "my dog died",
                               "my cat died", "pet is gone", "put to sleep", "euthanasia",
                               "put down", "grieving", "pet loss"]):
        return jsonify({"reply": "I'm so deeply sorry for your loss 💔 Losing a pet is heartbreaking — they are family. Please be gentle with yourself and give yourself time to grieve. Your pet was lucky to have someone who loved them so much. 🌈🐾"})

    # ---- WHAT CAN YOU DO ----
    if any(w in msg for w in ["what can you help with", "what do you know", "what can i ask",
                               "topics you cover", "what topics", "capabilities", "your skills",
                               "what are you good at", "help topics", "what can i talk about"]):
        return jsonify({"reply": "Here's what I can help you with 🤖: 🐾 Pet symptoms & diseases | 🐶 Dog & cat health | 🔬 How to use the prediction tool | 💊 Treatment & prevention | 🍖 Diet & nutrition | 🏥 When to visit a vet | 🛁 Grooming & hygiene | 💉 Vaccinations & parasites | 😟 Pet stress & behaviour | 🆕 New pet tips | 📬 Contacting our team. Just ask away!"})

    # ---- YES ----
    if msg in ["yes", "yeah", "yep", "yup", "sure", "of course", "absolutely",
               "definitely", "ok", "okay", "alright", "yea"]:
        return jsonify({"reply": "Great! 😊 Let me know what you need help with — symptoms, predictions, or anything about your pet!"})

    # ---- NO ----
    if msg in ["no", "nope", "nah", "not really", "no thanks", "nah thanks"]:
        return jsonify({"reply": "No worries! 😊 I'm here whenever you need me. Feel free to ask anything about your pet's health anytime!"})

    # ---- DEFAULT FALLBACK ----
    return jsonify({"reply": "Hmm, I'm not sure I understood that 🤔 I can help with pet symptoms, disease predictions, health tips, diet, grooming, vaccinations, and how to use this site. Try typing 'what can you help with' to see all topics!"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
