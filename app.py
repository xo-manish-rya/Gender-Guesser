import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify
import io
import os
from werkzeug.utils import secure_filename
import re
from collections import Counter
import nltk
from nltk.corpus import names
import gender_guesser.detector as gender_detector
from names_dataset import NameDataset

# Download required NLTK data
nltk.data.path.append('/tmp/nltk_data')
try:
    nltk.data.find('corpora/names')
except LookupError:
    nltk.download('names', download_dir='/tmp/nltk_data')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
# For Render deployment, use /tmp directory
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize gender detector and name dataset
detector = gender_detector.Detector()
name_dataset = NameDataset()

def fix_column_names(df):
    """Automatically detect and normalize column names for 'username' and 'fullnames'."""

    # Lowercase and clean column names (remove special chars and spaces)
    cleaned_columns = {
        col: re.sub(r'[^a-z]', '', col.lower())
        for col in df.columns
    }

    username_keywords = ['username', 'user', 'userid', 'instagramusername', 'igusername']
    fullname_keywords = ['fullname', 'name', 'fullnames', 'fullnameuser', 'igname']

    username_col = None
    fullname_col = None

    # Try to find the most likely username and fullname columns
    for original, cleaned in cleaned_columns.items():
        if any(keyword in cleaned for keyword in username_keywords):
            username_col = original
        elif any(keyword in cleaned for keyword in fullname_keywords):
            fullname_col = original

    # Rename detected columns
    if username_col:
        df.rename(columns={username_col: 'username'}, inplace=True)
    if fullname_col:
        df.rename(columns={fullname_col: 'fullnames'}, inplace=True)

    return df

class GenderAnalyzer:
    def __init__(self):
        # NLTK English names
        self.male_names = set(n.lower() for n in names.words('male.txt'))
        self.female_names = set(n.lower() for n in names.words('female.txt'))

        # Initialize name dataset
        self.name_dataset = NameDataset()
        
        # Cache for name dataset results to improve performance
        self.name_cache = {}

        # Basic regional databases as fallback
        self.indian_male = self._load_indian_male()
        self.indian_female = self._load_indian_female()
        
        self.nickname_map = {
            'alex': 'male', 'ali': 'male', 'robo': 'male', 'sandy': 'female',
            'kat': 'female', 'nick': 'male', 'sam': 'unknown',
            'kris': 'unknown', 'ramu': 'male', 'ammu': 'female',
            # Additional common nicknames
            'chris': 'male', 'jess': 'female', 'robin': 'unknown',
            'stacey': 'female', 'jordan': 'unknown', 'taylor': 'unknown',
            'casey': 'unknown', 'jamie': 'unknown', 'danny': 'male',
            'franky': 'male', 'bobby': 'male', 'rick': 'male',
            'tony': 'male', 'mike': 'male', 'steve': 'male', 'dave': 'male',
            'beth': 'female', 'liz': 'female', 'kate': 'female', 'meg': 'female'
        }

        self.detector = detector

        # Country-specific name endings and patterns
        self.country_patterns = {
            # Arabic/Middle Eastern
            'arabic': {
                'male_endings': ['d', 'r', 'm', 'n', 'f', 's', 't'],
                'female_endings': ['a', 'ah', 'ia', 'ya', 'na', 'ma', 'ra'],
                'common_suffixes': ['al-', 'el-', 'bin ', 'bint ']
            },
            # Spanish/Latin American
            'spanish': {
                'male_endings': ['o', 'os', 'n', 'r', 'l', 's'],
                'female_endings': ['a', 'as', 'ia', 'ra', 'na', 'la'],
                'common_suffixes': ['de ', 'del ', 'y ']
            },
            # Indian/South Asian
            'indian': {
                'male_endings': ['esh', 'esh', 'ant', 'pal', 'raj', 'jit', 'bir', 'deep', 'inder'],
                'female_endings': ['a', 'i', 'aa', 'ti', 'ni', 'li', 'ya', 'ika', 'priya'],
                'common_suffixes': ['Kumar', 'Singh', 'Devi']
            },
            # East Asian
            'asian': {
                'male_endings': ['wei', 'jun', 'hao', 'qiang', 'min', 'feng'],
                'female_endings': ['mei', 'ling', 'fang', 'yan', 'hui', 'jing'],
                'common_suffixes': []
            },
            # African
            'african': {
                'male_endings': ['su', 'ku', 'ba', 'ma', 'fa', 'ka'],
                'female_endings': ['na', 'ma', 'ta', 'ra', 'sha'],
                'common_suffixes': ['Olu', 'Ade', 'Nne']
            },
            # Slavic
            'slavic': {
                'male_endings': ['v', 'n', 'r', 'y', 'k', 'l', 's'],
                'female_endings': ['a', 'ya', 'ia', 'na', 'ka', 'ra'],
                'common_suffixes': ['ova', 'ev', 'in', 'sky', 'ski']
            },
            # Scandinavian
            'scandinavian': {
                'male_endings': ['r', 'n', 'd', 's', 't', 'l'],
                'female_endings': ['a', 'e', 'n', 'r', 'l'],
                'common_suffixes': ['sen', 'sson', 'dottir']
            }
        }

    def _load_indian_male(self):
        return {
            'arjun','rahul','rohit','aman','arvind','karan','prashant','sunil',
            'ankit','sumit','nitin','sachin','rajesh','vikas','deepak','vinay',
            'sandeep','rakesh','shivam','tarun','yash','krishna','aryan','sagar',
            'vishal','manish','pankaj','rajan','suresh','naveen','gaurav','harish',
            'mahesh','dinesh','rajiv','amitabh','vikram','aditya','siddharth',
            'pranav','abhishek','chetan','darshan','farhan','girish','hemant'
        }

    def _load_indian_female(self):
        return {
            'priya','neha','anjali','kavita','pooja','riya','tanya','megha',
            'sakshi','isha','aarti','anita','sonam','swati','divya','shruti',
            'radha','mansi','jaya','kriti','nandini','shreya','preeti','monika',
            'pallavi','sneha','urvashi','vasudha','yamini','zoya','kiran','madhu',
            'bhavna','chitra','dipika','ekta','falguni','gayatri','hina','indira'
        }

    def query_name_dataset(self, first_name):
        """Query the name-dataset for gender information with caching"""
        if first_name in self.name_cache:
            return self.name_cache[first_name]
        
        try:
            # Capitalize the name as the dataset expects proper casing
            name_capitalized = first_name.capitalize()
            result = self.name_dataset.search(name_capitalized)
            
            if result and len(result) > 0:
                # Get the first result (most common)
                name_info = list(result.values())[0]
                gender = name_info.get('gender')
                country = name_info.get('country')
                
                # Convert dataset gender to our format
                if gender == 'M':
                    gender_result = 'male'
                elif gender == 'F':
                    gender_result = 'female'
                else:
                    gender_result = 'unknown'
                
                # Handle country data - it might be a dictionary or string
                country_code = None
                if isinstance(country, dict):
                    # If country is a dictionary, get the country code from it
                    country_code = country.get('country_code')
                elif isinstance(country, str):
                    country_code = country
                
                result_data = {
                    'gender': gender_result,
                    'country': country_code,
                    'confidence': 0.9  # High confidence for dataset results
                }
                
                self.name_cache[first_name] = result_data
                return result_data
                
        except Exception as e:
            print(f"Error querying name dataset for {first_name}: {e}")
        
        # Return unknown if no result or error
        self.name_cache[first_name] = {'gender': 'unknown', 'country': None, 'confidence': 0}
        return {'gender': 'unknown', 'country': None, 'confidence': 0}

    def detect_language_pattern(self, name):
        """Enhanced language detection based on name patterns and country data"""
        name_lower = name.lower()
        
        # First, try to get country info from name dataset
        dataset_result = self.query_name_dataset(name_lower)
        if dataset_result['country']:
            country = dataset_result['country']
            
            # Ensure country is a string before calling upper()
            if isinstance(country, str):
                country_upper = country.upper()
                
                # Map countries to language/region groups
                country_to_region = {
                    # Arabic speaking countries
                    'SA': 'arabic', 'AE': 'arabic', 'EG': 'arabic', 'IQ': 'arabic',
                    'IR': 'arabic', 'JO': 'arabic', 'LB': 'arabic', 'SY': 'arabic',
                    'YE': 'arabic', 'OM': 'arabic', 'QA': 'arabic', 'KW': 'arabic',
                    'BH': 'arabic',
                    
                    # Spanish speaking countries
                    'ES': 'spanish', 'MX': 'spanish', 'AR': 'spanish', 'CO': 'spanish',
                    'CL': 'spanish', 'PE': 'spanish', 'VE': 'spanish', 'EC': 'spanish',
                    'GT': 'spanish', 'CU': 'spanish', 'BO': 'spanish', 'DO': 'spanish',
                    'HN': 'spanish', 'PY': 'spanish', 'SV': 'spanish', 'NI': 'spanish',
                    'CR': 'spanish', 'PR': 'spanish', 'PA': 'spanish', 'UY': 'spanish',
                    
                    # Indian subcontinent
                    'IN': 'indian', 'PK': 'indian', 'BD': 'indian', 'NP': 'indian',
                    'LK': 'indian',
                    
                    # East Asian
                    'CN': 'asian', 'JP': 'asian', 'KR': 'asian', 'TW': 'asian',
                    'HK': 'asian', 'SG': 'asian',
                    
                    # African regions
                    'NG': 'african', 'ET': 'african', 'EG': 'african', 'CD': 'african',
                    'ZA': 'african', 'TZ': 'african', 'KE': 'african', 'UG': 'african',
                    'DZ': 'african', 'SD': 'african', 'MA': 'african', 'AO': 'african',
                    'MZ': 'african', 'GH': 'african',
                    
                    # Slavic regions
                    'RU': 'slavic', 'UA': 'slavic', 'BY': 'slavic', 'PL': 'slavic',
                    'CZ': 'slavic', 'SK': 'slavic', 'RS': 'slavic', 'BG': 'slavic',
                    'HR': 'slavic', 'SI': 'slavic',
                    
                    # Scandinavian
                    'SE': 'scandinavian', 'NO': 'scandinavian', 'DK': 'scandinavian',
                    'FI': 'scandinavian', 'IS': 'scandinavian'
                }
                
                region = country_to_region.get(country_upper)
                if region:
                    return region
        
        # Fallback to character pattern detection
        if re.search(r'[\u0600-\u06FF]', name):  # Arabic characters
            return 'arabic'
        elif re.search(r'[\u0400-\u04FF]', name):  # Cyrillic characters
            return 'slavic'
        elif re.search(r'[\u4E00-\u9FFF]', name):  # Chinese characters
            return 'asian'
        elif re.search(r'[\u3040-\u309F\u30A0-\u30FF]', name):  # Japanese characters
            return 'asian'
        elif re.search(r'[\uAC00-\uD7AF]', name):  # Korean characters
            return 'asian'
            
        return 'english'

    def clean_name(self, name):
        if pd.isna(name) or not isinstance(name, str):
            return ""
        name = name.lower().strip()

        name = re.sub(r'[^a-z\s]', '', name)
        parts = name.split()
        if not parts:
            return ""

        # Remove titles and honorifics
        titles = {'mr','mrs','ms','miss','dr','sir','madam','prof','mx',
                 'lord','lady','dame','rev','fr','br','sr','jr','esq'}
        if parts[0] in titles and len(parts) > 1:
            return parts[1]

        return parts[0]

    def analyze_username_pattern(self, username):
        if pd.isna(username) or not isinstance(username, str):
            return "unknown"

        username = username.lower()

        # Enhanced pattern matching with comprehensive word lists
        female_words = [
            'princess','queen','girl','cutie','baby','doll','babe',
            'beauty','lovely','sweet','pretty','goddess','miss','lady',
            'femme','pink','angel','rose','lily','barbie','diva','chic',
            'butterfly','flower','sparkle','glam','girly','sis','sister',
            'mama','wife','bride','blossom','pearl','ruby','daisy',
            'venus','aphrodite','cinderella','snowwhite','aurora'
        ]
        
        male_words = [
            'king','boy','bro','dude','boss','man','warrior','titan',
            'alpha','sigma','mr','sir','lord','gentleman','gamer','tech',
            'beast','master','captain','giant','wolf','lion','alpha',
            'beta','gamma','delta','epsilon','stud','chad','brother',
            'papa','dad','husband','groom','knight','samurai','ninja',
            'zeus','apollo','hercules','thor','odin','anubis'
        ]
        
        # Enhanced regex patterns
        female_patterns = [
            r'.*princess.*', r'.*queen.*', r'.*girl.*', r'.*barbie.*', 
            r'.*doll.*', r'.*pink.*', r'.*lovely.*', r'.*cutie.*',
            r'.*beauty.*', r'.*angel.*', r'.*rose.*', r'.*lily.*',
            r'.*sweet.*', r'.*pretty.*', r'.*goddess.*', r'.*miss_.*',
            r'.*lady.*', r'.*femme.*', r'.*she_.*', r'.*her_.*',
            r'.*mrs_.*', r'.*ms_.*', r'.*bride.*', r'.*wife.*',
            r'.*daughter.*', r'.*sister.*', r'.*mama.*', r'.*mom.*'
        ]
        
        male_patterns = [
            r'.*king.*', r'.*boy.*', r'.*guy.*', r'.*bro.*', r'.*dude.*',
            r'.*man.*', r'.*mr_.*', r'.*sir.*', r'.*lord.*', r'.*boss.*',
            r'.*beast.*', r'.*titan.*', r'.*warrior.*', r'.*he_.*',
            r'.*him_.*', r'.*gentleman.*', r'.*alpha.*', r'.*sigma.*',
            r'.*beta.*', r'.*gamma.*', r'.*stud.*', r'.*chad.*',
            r'.*papa.*', r'.*dad.*', r'.*husband.*', r'.*groom.*',
            r'.*brother.*', r'.*son.*', r'.*father.*'
        ]

        # Simple word matching with weighting
        female_score = sum(2 for w in female_words if w in username)
        male_score = sum(2 for w in male_words if w in username)
        
        # Regex pattern matching
        female_score += sum(3 for pattern in female_patterns if re.match(pattern, username))
        male_score += sum(3 for pattern in male_patterns if re.match(pattern, username))

        # Additional pattern: numbers at the end (often used by males)
        if re.search(r'\d+$', username):
            male_score += 1

        if female_score > male_score:
            return "female"
        elif male_score > female_score:
            return "male"

        return "unknown"

    def apply_country_specific_rules(self, first_name, language):
        """Apply country-specific linguistic rules to determine gender"""
        score_m = score_f = 0
        
        if language in self.country_patterns:
            patterns = self.country_patterns[language]
            
            # Check endings
            for ending in patterns['male_endings']:
                if first_name.endswith(ending):
                    score_m += 0.5
                    
            for ending in patterns['female_endings']:
                if first_name.endswith(ending):
                    score_f += 0.5
            
            # Check common suffixes
            for suffix in patterns['common_suffixes']:
                if suffix in first_name:
                    if language == 'indian' and suffix in ['Kumar', 'Singh']:
                        score_m += 1
                    elif language == 'indian' and suffix == 'Devi':
                        score_f += 1
                    elif language == 'scandinavian' and suffix == 'dottir':
                        score_f += 1
                    elif language == 'scandinavian' and suffix in ['sen', 'sson']:
                        score_m += 1
        
        return score_m, score_f

    def predict_gender(self, fullname, username):
        first = self.clean_name(fullname)
        if not first:
            return self.analyze_username_pattern(username)

        # Priority 1: Query the name-dataset (most reliable)
        dataset_result = self.query_name_dataset(first)
        if dataset_result['gender'] != 'unknown':
            return dataset_result['gender']

        # Detect language pattern
        language = self.detect_language_pattern(first)

        score_m = score_f = 0

        # Priority 2: nickname map
        if first in self.nickname_map:
            if self.nickname_map[first] == 'male': score_m += 1
            elif self.nickname_map[first] == 'female': score_f += 1

        # Priority 3: Regional name databases
        if first in self.indian_male: score_m += 1.2
        if first in self.indian_female: score_f += 1.2

        # Priority 4: NLTK English names
        if first in self.male_names: score_m += 0.7
        if first in self.female_names: score_f += 0.7

        # Priority 5: gender-guesser
        try:
            g = self.detector.get_gender(first.capitalize())
            if g in ['male','mostly_male']: score_m += 0.6
            if g in ['female','mostly_female']: score_f += 0.6
        except:
            pass

        # Priority 6: Country-specific linguistic rules
        country_m, country_f = self.apply_country_specific_rules(first, language)
        score_m += country_m
        score_f += country_f

        # Priority 7: Username patterns
        username_result = self.analyze_username_pattern(username)
        if username_result == 'male': score_m += 0.4
        if username_result == 'female': score_f += 0.4

        # Bonus for non-English names
        if language != 'english':
            if score_m > 0: score_m += 0.3
            if score_f > 0: score_f += 0.3

        # Final decision
        if score_m > score_f and score_m >= 0.5:
            return "male"
        elif score_f > score_m and score_f >= 0.5:
            return "female"
        else:
            return username_result if username_result != 'unknown' else "unknown"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith(('.xlsx', '.xls', '.csv')):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read the file
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            # Auto-fix column names before checking
            df = fix_column_names(df)
            
            # Check required columns after fixing
            if 'username' not in df.columns or 'fullnames' not in df.columns:
                return jsonify({
                    'error': f'File must contain "username" and "fullnames" columns. Found columns: {list(df.columns)}'
                }), 400
            
            # Initialize analyzer
            analyzer = GenderAnalyzer()
            
            # Analyze genders
            genders = []
            confidence_scores = []
            countries = []
            
            for _, row in df.iterrows():
                gender = analyzer.predict_gender(row['fullnames'], row['username'])
                genders.append(gender)
                
                # Get additional info from name dataset
                first_name = analyzer.clean_name(row['fullnames'])
                dataset_result = analyzer.query_name_dataset(first_name)
                
                # Handle country data properly
                country_data = dataset_result['country']
                if isinstance(country_data, dict):
                    # Extract country code from dictionary
                    country_code = country_data.get('country_code', 'Unknown')
                else:
                    country_code = country_data if country_data else 'Unknown'
                
                countries.append(country_code)
                
                # Enhanced confidence scoring
                if dataset_result['gender'] != 'unknown':
                    confidence = 0.95  # Very high confidence for dataset results
                elif gender != 'unknown':
                    confidence = 0.8
                else:
                    confidence = 0.3
                confidence_scores.append(confidence)
            
            # Add results to dataframe
            df['gender'] = genders
            df['confidence'] = confidence_scores
            df['detected_country'] = countries
            
            # Generate statistics
            gender_counts = Counter(genders)
            total = len(genders)
            known_genders = total - gender_counts.get('unknown', 0)
            accuracy_rate = round((known_genders / total) * 100, 2) if total > 0 else 0
            
            stats = {
                'male': gender_counts.get('male', 0),
                'female': gender_counts.get('female', 0),
                'unknown': gender_counts.get('unknown', 0),
                'total': total,
                'male_percentage': round((gender_counts.get('male', 0) / total) * 100, 2),
                'female_percentage': round((gender_counts.get('female', 0) / total) * 100, 2),
                'unknown_percentage': round((gender_counts.get('unknown', 0) / total) * 100, 2),
                'accuracy_rate': accuracy_rate
            }
            
            # Save analyzed file
            output_filename = f"analyzed_{filename.split('.')[0]}.xlsx"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            df.to_excel(output_path, index=False)
            
            # Prepare sample data for display
            sample_data = df.head(10).to_dict('records')
            
            return jsonify({
                'success': True,
                'stats': stats,
                'sample_data': sample_data,
                'download_url': f'/download/{output_filename}',
                'message': f'✅ Analysis complete! Gender detection rate: {accuracy_rate}%'
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file format. Please upload Excel or CSV file.'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(app.config['UPLOAD_FOLDER'], filename),
        as_attachment=True,
        download_name=filename
    )

# Remove the existing app.run() and replace with:
if __name__ == '__main__':
    print("🚀 Supercharged Instagram Gender Analyzer is running!")
    print("📚 Powered by name-dataset with 160M+ names!")
    print("🌍 Now with country detection and higher accuracy!")
    print("📊 Access your app at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)