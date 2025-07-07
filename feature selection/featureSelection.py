import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, roc_auc_score, confusion_matrix)
from deap import base, creator, tools, algorithms
import shap
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    page_title="GA Feature Selector",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        padding: 1.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #4CAF50;
    }
    .st-ax {
        background-color: #f63366;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .feature-card {
        background: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class GeneticFeatureSelector:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.best_features = None
        self.best_model = None
        self.history = []
        self.models = {
            "Random Forest": RandomForestClassifier,
            "Gradient Boosting": GradientBoostingClassifier,
            "SVM": SVC
        }
        
    def load_data(self, file_path):
        try:
            if file_path.name.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.name.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            return True, "Data loaded successfully!"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def preprocess_data(self, target_column):
        try:
            # Separate features and target
            X = self.data.drop(target_column, axis=1)
            y = self.data[target_column]
            
            # Encode categorical variables
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            if not categorical_cols.empty:
                le = LabelEncoder()
                y = le.fit_transform(y)
                X = pd.get_dummies(X, columns=categorical_cols)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.3, random_state=42)
            
            # Scale numerical features
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
            if not numerical_cols.empty:
                self.X_train[numerical_cols] = self.scaler.fit_transform(self.X_train[numerical_cols])
                self.X_test[numerical_cols] = self.scaler.transform(self.X_test[numerical_cols])
            
            return True, "Data preprocessed successfully!"
        except Exception as e:
            return False, f"Error preprocessing data: {str(e)}"
    
    def evaluate_features(self, individual, model_class):
        mask = np.array(individual, dtype=bool)
        if sum(mask) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        X_train_selected = self.X_train.iloc[:, mask]
        X_test_selected = self.X_test.iloc[:, mask]

        model = model_class(random_state=42)
        model.fit(X_train_selected, self.y_train)

        y_pred = model.predict(X_test_selected)
        y_proba = model.predict_proba(X_test_selected) if hasattr(model, "predict_proba") else None

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')

        # ROC AUC handling for multiclass
        try:
            if y_proba is not None:
                if len(np.unique(self.y_test)) > 2:
                    roc_auc = roc_auc_score(self.y_test, y_proba, multi_class='ovr', average='weighted')
                else:
                    roc_auc = roc_auc_score(self.y_test, y_proba[:, 1])
            else:
                roc_auc = 0.5
        except Exception:
            roc_auc = 0.5

        return accuracy, f1, precision, recall, roc_auc
    
    def run_genetic_algorithm(self, model_name, population_size=50, generations=20, 
                             crossover_prob=0.5, mutation_prob=0.2, tournament_size=3):
        try:
            n_features = self.X_train.shape[1]
            model_class = self.models[model_name]
            
            # Create fitness and individual classes
            creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            # Initialize toolbox
            toolbox = base.Toolbox()
            toolbox.register("attr_bool", np.random.randint, 0, 2)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            def evaluate(individual):
                acc, f1, prec, rec, _ = self.evaluate_features(individual, model_class)
                return (acc, f1, prec, rec)
            
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
            toolbox.register("select", tools.selTournament, tournsize=tournament_size)
            
            # Create population
            pop = toolbox.population(n=population_size)
            
            # Statistics
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run algorithm with progress updates
            for gen in range(generations):
                pop, logbook = algorithms.eaSimple(
                    pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, 
                    ngen=1, stats=stats, verbose=False)
                
                # Update progress
                progress = (gen + 1) / generations
                progress_bar.progress(progress)
                status_text.text(f"Generation {gen+1}/{generations} - Max Fitness: {logbook[-1]['max']:.4f}")
                
                # Store history
                self.history.extend(logbook)
            
            # Get best individual
            best_individual = tools.selBest(pop, k=1)[0]
            self.best_features = np.array(best_individual, dtype=bool)
            self.selected_features = self.X_train.columns[self.best_features]
            
            # Train final model
            X_train_selected = self.X_train.iloc[:, self.best_features]
            X_test_selected = self.X_test.iloc[:, self.best_features]
            
            self.best_model = model_class(random_state=42)
            self.best_model.fit(X_train_selected, self.y_train)
            
            progress_bar.empty()
            status_text.empty()
            
            return True, "Genetic algorithm completed successfully!"
        except Exception as e:
            return False, f"Error running genetic algorithm: {str(e)}"
    
    def get_performance_metrics(self):
        if self.best_model is None:
            return None

        X_test_selected = self.X_test.iloc[:, self.best_features]
        y_pred = self.best_model.predict(X_test_selected)
        y_proba = self.best_model.predict_proba(X_test_selected) if hasattr(self.best_model, "predict_proba") else None

        # ROC AUC handling for multiclass
        try:
            if y_proba is not None:
                if len(np.unique(self.y_test)) > 2:
                    roc_auc = roc_auc_score(self.y_test, y_proba, multi_class='ovr', average='weighted')
                else:
                    roc_auc = roc_auc_score(self.y_test, y_proba[:, 1])
            else:
                roc_auc = "N/A"
        except Exception:
            roc_auc = "N/A"

        metrics = {
            "Accuracy": accuracy_score(self.y_test, y_pred),
            "F1 Score": f1_score(self.y_test, y_pred, average='weighted'),
            "Precision": precision_score(self.y_test, y_pred, average='weighted'),
            "Recall": recall_score(self.y_test, y_pred, average='weighted'),
            "ROC AUC": roc_auc,
            "Num Features": sum(self.best_features),
            "Feature Reduction": f"{((len(self.best_features) - sum(self.best_features))) / len(self.best_features) * 100:.1f}%"
        }

        return metrics, y_pred, y_proba
    
    def explain_model(self):
        if self.best_model is None:
            return None
        
        X_train_selected = self.X_train.iloc[:, self.best_features]
        
        # Feature importance
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': self.selected_features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
        else:
            importance_df = None
        
        # SHAP values
        try:
            explainer = shap.Explainer(self.best_model, X_train_selected)
            shap_values = explainer(X_train_selected)
            return importance_df, shap_values
        except:
            return importance_df, None

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

def plot_roc_curve(y_true, y_proba):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    return fig

def get_image_download_link(fig, filename="plot.png", text="Download Plot"):
    """Generates a link allowing the plot to be downloaded"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{data}" download="{filename}">{text}</a>'
    return href

def main():
    st.title("🧬 Genetic Algorithm Feature Selection")
    st.markdown("""
    This app uses Genetic Algorithms to select the most relevant features for your machine learning model.
    Upload your dataset, select the target variable, and configure the GA parameters to get started.
    """)
    
    # Initialize session state
    if 'selector' not in st.session_state:
        st.session_state.selector = GeneticFeatureSelector()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Input Parameters")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx', 'xls'])
        if uploaded_file is not None:
            success, message = st.session_state.selector.load_data(uploaded_file)
            if success:
                st.success(message)
            else:
                st.error(message)
        
        # Target selection
        if st.session_state.selector.data is not None:
            target_col = st.selectbox(
                "Select Target Variable",
                options=st.session_state.selector.data.columns
            )
            
            if st.button("Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    success, message = st.session_state.selector.preprocess_data(target_col)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        # GA parameters
        st.subheader("Genetic Algorithm Settings")
        model_name = st.selectbox(
            "Model",
            options=list(st.session_state.selector.models.keys())
        )
        
        col1, col2 = st.columns(2)
        pop_size = col1.number_input("Population Size", min_value=10, max_value=500, value=50)
        generations = col2.number_input("Generations", min_value=5, max_value=100, value=20)
        
        col1, col2 = st.columns(2)
        cx_prob = col1.slider("Crossover Probability", 0.0, 1.0, 0.5)
        mut_prob = col2.slider("Mutation Probability", 0.0, 1.0, 0.2)
        
        tournament_size = st.slider("Tournament Size", 2, 10, 3)
        
        if st.button("Run Feature Selection"):
            if st.session_state.selector.X_train is None:
                st.error("Please preprocess the data first")
            else:
                with st.spinner("Running genetic algorithm..."):
                    success, message = st.session_state.selector.run_genetic_algorithm(
                        model_name, pop_size, generations, cx_prob, mut_prob, tournament_size)
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    # Main content area
    if st.session_state.selector.data is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.selector.data.head())
        
        if st.session_state.selector.X_train is not None:
            st.info(f"Training samples: {len(st.session_state.selector.X_train)} | Test samples: {len(st.session_state.selector.X_test)}")
    
    # Results section
    if st.session_state.selector.best_model is not None:
        st.header("Results")
        
        # Performance metrics
        metrics, y_pred, y_proba = st.session_state.selector.get_performance_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        col2.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
        col3.metric("Precision", f"{metrics['Precision']:.4f}")
        col4.metric("Recall", f"{metrics['Recall']:.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Selected Features", metrics['Num Features'])
        with col2:
            st.metric("Feature Reduction", metrics['Feature Reduction'])
        
        # Selected features
        st.subheader("Selected Features")
        st.write(list(st.session_state.selector.selected_features))
        
        # Visualizations
        st.header("Visualizations")
        
        # Performance metrics plot
        st.subheader("Model Performance")
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and k not in ['Num Features']}
        ax.bar(plot_metrics.keys(), plot_metrics.values())
        ax.set_ylim(0, 1)
        for i, v in enumerate(plot_metrics.values()):
            ax.text(i, v + 0.02, f"{v:.3f}", ha='center')
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "performance_metrics.png"), unsafe_allow_html=True)
        
        # Confusion Matrix and ROC Curve
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            cm_fig = plot_confusion_matrix(st.session_state.selector.y_test, y_pred)
            st.pyplot(cm_fig)
            st.markdown(get_image_download_link(cm_fig, "confusion_matrix.png"), unsafe_allow_html=True)
        
        with col2:
            if y_proba is not None:
                st.subheader("ROC Curve")
                roc_fig = plot_roc_curve(st.session_state.selector.y_test, y_proba)
                st.pyplot(roc_fig)
                st.markdown(get_image_download_link(roc_fig, "roc_curve.png"), unsafe_allow_html=True)
        
        # Feature Importance
        st.subheader("Feature Importance")
        importance_df, shap_values = st.session_state.selector.explain_model()
        
        if importance_df is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
            ax.set_title("Feature Importance")
            st.pyplot(fig)
            st.markdown(get_image_download_link(fig, "feature_importance.png"), unsafe_allow_html=True)
        
        # SHAP Summary Plot
        if shap_values is not None:
            st.subheader("SHAP Values")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, st.session_state.selector.X_train.iloc[:, st.session_state.selector.best_features], plot_type="bar", show=False)
            st.pyplot(fig)
            st.markdown(get_image_download_link(fig, "shap_summary.png"), unsafe_allow_html=True)
        
        # GA Convergence Plot
        st.subheader("GA Convergence")
        if st.session_state.selector.history:
            gen = [entry['gen'] for entry in st.session_state.selector.history]
            avg = [entry['avg'][0] for entry in st.session_state.selector.history]  # Using accuracy for convergence
            max_ = [entry['max'][0] for entry in st.session_state.selector.history]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(gen, avg, label="Average Fitness")
            ax.plot(gen, max_, label="Maximum Fitness")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness (Accuracy)")
            ax.set_title("GA Convergence")
            ax.legend()
            st.pyplot(fig)
            st.markdown(get_image_download_link(fig, "ga_convergence.png"), unsafe_allow_html=True)
        
        # Download results
        st.subheader("Download Results")
        if st.button("Export All Visualizations"):
            # Create a zip file with all visualizations
            import zipfile
            from io import BytesIO
            
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add figures to zip
                figures = {
                    "performance_metrics.png": fig,
                    "confusion_matrix.png": cm_fig,
                    "roc_curve.png": roc_fig if y_proba is not None else None,
                    "feature_importance.png": fig if importance_df is not None else None,
                    "shap_summary.png": fig if shap_values is not None else None,
                    "ga_convergence.png": fig if st.session_state.selector.history else None
                }
                
                for filename, fig_obj in figures.items():
                    if fig_obj is not None:
                        buf = BytesIO()
                        fig_obj.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        zip_file.writestr(filename, buf.getvalue())
                
                # Add selected features as CSV
                features_df = pd.DataFrame({
                    'Selected Features': list(st.session_state.selector.selected_features)
                })
                zip_file.writestr("selected_features.csv", features_df.to_csv(index=False))
                
                # Add performance metrics as CSV
                metrics_df = pd.DataFrame([metrics])
                zip_file.writestr("performance_metrics.csv", metrics_df.to_csv(index=False))
            
            st.download_button(
                label="Download All Results",
                data=zip_buffer.getvalue(),
                file_name="ga_feature_selection_results.zip",
                mime="application/zip"
            )

if __name__ == "__main__":
    main()