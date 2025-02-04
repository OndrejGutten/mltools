import numpy as np
from sklearn.metrics import accuracy_score
import xgboost
from sklearn.feature_extraction.text import TfidfVectorizer

# Class weights
dummy_class_weights = np.array([10, 5, 1])
def weighted_accuracy(y_true, y_pred, class_weights):
    sample_weights = class_weights[y_true]
    return np.sum(  np.array(y_true == y_pred) * sample_weights) / np.sum(sample_weights)

def weighted_probability(y_true, y_pred, class_weights):
    sample_weights = class_weights[y_true]
    return np.sum(  np.array(y_pred[np.arange(y_pred.shape[0]),y_true]) * sample_weights) / np.sum(sample_weights)

def weighted_accuracy_loss(class_weights):
    class_weights = np.array(class_weights) # allows indexing by labels
    def custom_loss(preds, dtrain):
        labels = dtrain.get_label().astype(int)
        preds = preds.reshape(-1, len(class_weights))
        preds = np.clip(preds, 1e-7, 1 - 1e-7)  # Avoid numerical issues

        # One-hot encoding of labels
        one_hot = np.zeros_like(preds)
        one_hot[np.arange(len(labels)), labels] = 1

        # Weighted log loss
        grad = -class_weights[labels][:, None] * (one_hot - preds)
        hess = class_weights[labels][:, None] * preds * (1 - preds)

        return grad.flatten(), hess.flatten()

    return custom_loss

# Simulated multi-class data
import xgboost
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create synthetic dataset with 3 classes
X, y = make_classification(
    n_samples=10000, n_features=20, n_classes=3, n_informative=10, random_state=42
)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to DMatrix format
dtrain_dummy = xgboost.DMatrix(X_train, label=y_train)
dtest_dummy = xgboost.DMatrix(X_test, label=y_test)

# Set up xgboost parameters

params = {
    "objective": "multi:softprob",
    "num_class": len(dummy_class_weights),
    "lambda": 0,  # Disable L2 regularization
    "alpha": 0    # Disable L1 regularization
}

weights = np.array([10,5,1])
custom_loss = weighted_accuracy_loss(weights)
bst = xgboost.train(
    params,
    dtrain_dummy,
    num_boost_round=100,
    obj=custom_loss,
    evals=[(dtest_dummy, "test")],
    early_stopping_rounds=10
)
y_pred = np.argmax(bst.predict(dtest_dummy),axis=1)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Weighted accuracy: {weighted_accuracy(y_test, y_pred, dummy_class_weights)}")
print(f"Weighted probability: {weighted_probability(y_test, bst.predict(dtest_dummy), dummy_class_weights)}")


### this does not seem to produce best results for given class weights - even on the training set
acc = np.zeros((4,4,4))
wacc = np.zeros((4,4,4))
wp = np.zeros((4,4,4))
vals = [1,3,5,10]
for w1 in range(len(vals)):
    for w2 in range(len(vals)):
        for w3 in range(len(vals)):
            weights = np.array([vals[w1],vals[w2],vals[w3]])
            custom_loss = sum_of_probabilities_loss(weights)
            bst = xgboost.train(
     params,
    dtrain,
    num_boost_round=100,
    obj=custom_loss,
    evals=[(dtest, "test")],
    early_stopping_rounds=10
)   
            y_pred = np.argmax(bst.predict(dtrain),axis=1)
            acc[w1,w2,w3] = accuracy_score(y_train, y_pred)
            wacc[w1,w2,w3] = weighted_accuracy(y_train,y_pred, class_weights)
            wp[w1,w2,w3] = weighted_probability(y_train, bst.predict(dtrain),class_weights)

### bst.train does not run with text data and i dont understand why. Here we investigate
X_train_text = dss[0].df['data']
y_train_text = dss[0].df['label']
X_test_text = dss[1].df['data']
y_test_text = dss[1].df['label']
label_encoder = LabelEncoder()
transformed_y = label_encoder.fit_transform(y_train_text)

vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X_train_text)

dtrain_text = xgboost.DMatrix(X_vect,label = transformed_y)

dummy_weights = np.array(len(np.unique(transformed_y)) * [1])
custom_loss_text = weighted_accuracy_loss(dummy_weights)

params_text = {
    "objective": "multi:softprob",
    "num_class": len(dummy_weights),
}
# this does not work
bst_text = xgboost.train(
    params = params_text,
    dtrain = dtrain_text,
    num_boost_round=100,
    obj=custom_loss_text,
)


# TAKEAWAYS:
# label encoder necessary (labels in range of 0-N)
# class weights needs to be an array, not dictionary
# parameters['objective'] = 'multi:softprob', otherwise prediction is not probas for some reason
# num_classes given in params - this was the source of error

#### COST MATRIX ####
class_weights = np.array([1, 1, 1])  # Higher weight for class 2 and 3
cost_matrix = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

def custom_multiclass_loss(cost_matrix, class_weights):
    def custom_loss(preds,dtrain):
        # Reshape predictions to [n_samples, n_classes]
        num_classes = len(class_weights)
        preds = preds.reshape(-1, num_classes)
        labels = dtrain.get_label().astype(int)

        # Example class weights and cost matrix


        # Softmax for predicted probabilities
        preds = np.exp(preds) / np.exp(preds).sum(axis=1, keepdims=True)

        # Compute gradients and Hessians
        grad = np.zeros_like(preds)
        hess = np.zeros_like(preds)
        for i in range(len(labels)):
            true_class = labels[i]
            for j in range(num_classes):
                weight = class_weights[true_class] * (cost_matrix[true_class, j] + 1)
                grad[i, j] = weight * (preds[i, j] - (j == true_class))
                hess[i, j] = weight * preds[i, j] * (1 - preds[i, j])

        return grad.flatten(), hess.flatten()

    return custom_loss

#sort values of a pandas dataframe by largest off-diagonal elements and print their indices
mydf = pd.DataFrame(np.random.rand(5,5))
np.fill_diagonal(mydf.values,0)
