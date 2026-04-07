import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import os

def create_pipeline(model):
    """
    Creates a scikit-learn pipeline with standard scaling and the given model.
    This guarantees any new model can be passed in and trained with the same preprocessing steps.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cm_{model_name.replace(" ", "_")}.png'))
    plt.close()

def plot_roc_curves(roc_data, output_dir):
    plt.figure(figsize=(8, 6))
    for model_name, data in roc_data.items():
        plt.plot(data['fpr'], data['tpr'], label=f"{model_name} (AUC = {data['auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()

def main():
    data_path = 'e:/my project GSG/classification/data/diabetes.csv'
    output_dir = 'e:/my project GSG/classification/models'
    
    if not os.path.exists(output_dir):
         os.makedirs(output_dir)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    target_col = 'Outcome' if 'Outcome' in df.columns else df.columns[-1]
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    pipelines = {}
    for name, model in models.items():
        pipelines[name] = create_pipeline(model)
        
    roc_data = {}
    best_model_name = ""
    best_auc = 0
    best_acc = 0
    
    print("\n--- Model Evaluation ---")
    
    report_lines = []
    report_lines.append("# Model Evaluation Report\n")
    report_lines.append("## Overview\nThis report compares Logistic Regression, Random Forest, and SVM using a unified ML Pipeline.\n")
    report_lines.append("## Performance Metrics\n")
    
    for name, pipeline in pipelines.items():
        print(f"Training and Evaluating: {name}")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate Accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        report_lines.append(f"- **{name}**: Accuracy = {acc:.4f}")
        
        # Confusion Matrix
        plot_confusion_matrix(y_test, y_pred, name, output_dir)
        
        # ROC Curve data
        y_prob = pipeline.predict_proba(X_test)[:, 1]
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        report_lines[-1] += f" (AUC = {roc_auc:.4f})"
        
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model_name = name
            best_acc = acc
            
        print("-" * 30)
        
    plot_roc_curves(roc_data, output_dir)
    print(f"Saved evaluation plots to {output_dir}")
    
    report_lines.append("\n## Conclusion\n")
    report_lines.append(f"The best performing model is **{best_model_name}** with an AUC of {best_auc:.4f} and Accuracy of {best_acc:.4f}.")
    report_lines.append(f"It provides the best balance of True Positive Rate and False Positive Rate, as seen in the ROC curve and overall accuracy.\n")
    
    report_path = os.path.join(output_dir, 'evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"Generated evaluation report at: {report_path}")

if __name__ == "__main__":
    main()
