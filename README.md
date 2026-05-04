# 🤖 AI Health Insurance Advisor

A beautiful and intelligent Streamlit application for medical cost prediction using advanced machine learning.

## ✨ Features

- 🔐 **User Authentication**: Secure login/signup system with data persistence
- 🏥 **Medical Cost Prediction**: AI-powered predictions using Random Forest ML model
- 📊 **Health Analytics**: Comprehensive health profile analysis with color-coded metrics
- 🎯 **Personalized Suggestions**: Tailored health recommendations based on user data
- 💰 **Insurance Analysis**: Coverage calculation and provider recommendations
- 🖨️ **Print-Friendly Reports**: Professional reports that can be saved as PDF using Ctrl+P
- 🎨 **Beautiful UI**: Modern design with gradient backgrounds and responsive layout

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install streamlit pandas scikit-learn joblib reportlab
   ```

2. **Add Background Image (Optional)**:
   - Place any of these files in the same folder as the app:
     - `background.jpg`
     - `background.png`
     - `bg.jpg`
     - `bg.png`
   - The app will automatically detect and use the first available image
   - If no image is found, it uses a beautiful gradient background

3. **Run the Application**:
   ```bash
   streamlit run appp.py
   ```

## 📋 How to Use

1. **Sign Up**: Create a new account with patient ID, name, and password
2. **Login**: Use your credentials to access the system
3. **Enter Health Data**: Fill out your medical information in the organized form
4. **Generate Report**: Click "Generate Prediction Report" to see your personalized analysis
5. **Save as PDF**: Use `Ctrl+P` (or `Cmd+P` on Mac) to print/save as PDF

## 🏗️ Architecture

- **Frontend**: Streamlit with custom CSS and responsive design
- **Backend**: Python with scikit-learn for ML predictions
- **Data Processing**: Custom OOP pipeline for data cleaning and preprocessing
- **Storage**: JSON-based user management system
- **Styling**: CSS with gradient backgrounds and modern UI components

## 📊 Report Features

- **Health Profile Summary**: Organized display of all health metrics
- **Cost Prediction**: Prominent display of predicted annual medical costs
- **Personalized Recommendations**: AI-generated health suggestions
- **Insurance Analysis**: Coverage calculations and provider links
- **Print-Friendly Design**: Optimized for PDF generation

## 🎨 Customization

### Background Images
Add any image file with these names to customize the background:
- `background.jpg/png`
- `bg.jpg/png`

### Colors and Styling
Modify the `local_css()` function in `appp.py` to customize colors, fonts, and layout.

## 🔧 Technical Details

- **ML Model**: Random Forest Regressor trained on medical cost data
- **Data Processing**: Automated cleaning, encoding, and feature engineering
- **User Management**: Secure password hashing with JSON storage
- **PDF Generation**: Browser-native printing for professional reports

## 📈 Future Enhancements

- [ ] Additional ML models (Neural Networks, Gradient Boosting)
- [ ] Advanced health risk analysis
- [ ] Integration with real insurance APIs
- [ ] Multi-language support
- [ ] Mobile app version

---

**Built with ❤️ by Qasim using Streamlit and Machine Learning**