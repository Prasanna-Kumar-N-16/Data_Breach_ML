import os
from data_preprocessing import *
from visualization import *
from feature_engineering import *
import pandas as pd
import subprocess
from scipy.stats import chi2_contingency
import xgboost as xgb
from model_building import *
from sklearn.tree import export_graphviz
from xgboost import plot_tree

def perform_chi_square_test(df, feature):
    #perform Chi-square test for a categorical feature and the target variable "Records".
    chi2_statistic, chi2_p_value, _, _ = chi2_contingency(pd.crosstab(df[feature], df['Records']))
    print(f"Chi-Square Statistic for '{feature}':", chi2_statistic)
    print(f"Chi-Square p-value for '{feature}':", chi2_p_value)
    print('>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<')

def chi_square_test(df):
    perform_chi_square_test(df, 'Organization type')
    perform_chi_square_test(df, 'Method')
    perform_chi_square_test(df, 'Entity')
    perform_chi_square_test(df, 'Year')



def main():
    
    # Load data and do some pre processing
    df , df_heatmap = text_preprocessing("data/df_1.csv")
    if df is None:
        return
    # number of rows;
    print("Number of rows after pre processing : ",df.shape[0])

    # number of columns , axis=1 means column
    print("Number of columns after pre processing",df.shape[1])
    
    plot_save_heatmap(df_heatmap)
    
    chi_square_test(df)

    # x and y records
    df_heatmap.dropna(inplace=True)
    X = df_heatmap.drop('Records', axis=1)
    y=df_heatmap['Records'].shape
    print(X.shape), print(y)

    perform_pca(X)

    # Preprocess and split data
    train_X, test_X, train_y, test_y, data_dmatrix = preprocess_and_split_data(df, 'Records')

    # Train and evaluate models
    rf_reg = train_and_evaluate_models(train_X, test_X, train_y, test_y, data_dmatrix)

    # Get the 6th decision tree from the random forest model
    #tree = rf_reg.model.estimators_[5]

    # Export the decision tree to a Graphviz DOT file
    #export_graphviz(tree, out_file='tree.dot', feature_names=train_X.columns, rounded=True, precision=1)

    # Convert the DOT file to an image (assuming you have Graphviz installed)
    #subprocess.run(['dot',  'tree.dot',  'tree.png'])

    # Alternatively, if you're using XGBoost, you can plot the tree directly
    #fig, ax = plt.subplots(figsize=(30, 30))
    #plot_tree(tree, ax=ax)

    # Save the XGBoost tree as an image
   # plt.savefig(os.path.join(os.getcwd(), 'images', 'tree.png'))
    #plt.close()


    model, history = build_neural_network(train_X, train_y, test_X, test_y)

    history_dict = history.history
    loss_values = history_dict['mae']  # you can change this
    val_loss_values = history_dict['val_mae']  # you can also change this
    epochs = range(1, len(loss_values) + 1)  # range of X (no. of epochs)
    plt.plot(epochs, loss_values, 'bo', label='Training MAE')
    plt.plot(epochs, val_loss_values, 'orange', label='Validation MAE')
    plt.title('Training and validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), 'images', 'mae.png'))
    plt.close()


   

if __name__ == "__main__":
    main()
