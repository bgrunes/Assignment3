import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, \
    mean_squared_error, r2_score


# TODO: Dont forget to import the classifier/regressors

# Setup to train learning models
A = np.loadtxt('tictac_single.txt')
X_single = A[:, :9]  # Input features
y_single = A[:, 9:].ravel()  # Output labels

# X is a 3x3 grid for tictactoe. It should go like this:
#   x0  |   x1  |   x2
#   x3  |   x4  |   x5
#   x6  |   x7  |   x8

# Y gives the output of the game (the winner)

X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, test_size=0.2, random_state=42, stratify=y_single)

# **************SVM Model******************* single.txt
svm_clf_single = SVC(kernel='linear', C=0.1)
svm_clf_single.fit(X_train, y_train)

svm_y_pred = svm_clf_single.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_y_pred)
svm_cm = confusion_matrix(y_test, svm_y_pred)

print("*****************Linear SVM - Single*****************")
print(f'Accuracy: {svm_accuracy}')
print('Confusion Matrix:')
print(svm_cm)

svm_scores = cross_val_score(svm_clf_single, X_single, y_single, cv=10)
print(f'Cross-Validation Accuracy: {svm_scores.mean()}\n')

# ********************MLP Model**************************** single.txt
mlp_clf_single = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1750, random_state=42)

# Train the model
mlp_clf_single.fit(X_train, y_train)

mlp_y_pred = mlp_clf_single.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_y_pred)
mlp_cm = confusion_matrix(y_test, mlp_y_pred)

print("*****************MLP Model - Single*****************")
print(f'Accuracy: {mlp_accuracy}')
print('Confusion Matrix:')
print(mlp_cm)

mlp_scores = cross_val_score(mlp_clf_single, X_single, y_single, cv=10)
print(f'Cross-Validation Accuracy: {mlp_scores.mean()}\n')

# *****************KNN Model************************* single.txt
# Instantiate the kNN classifier with a specific value of k
knn_clf_single = KNeighborsClassifier(n_neighbors=5, metric='euclidean')  # Adjust the value of n_neighbors and metric as needed
# Standardize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_single)

# Split the scaled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_single, test_size=0.2, random_state=42)

# Train the Model
knn_clf_single.fit(X_train, y_train)
y_pred = knn_clf_single.predict(X_test)

print("*****************KNN Model - Single*****************")

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))

# Setup to train learning models **************** 2nd dataset
A = np.loadtxt('tictac_final.txt')
X_final = A[:, :9]   # Input features
y_final = A[:, 9:].ravel()   # Output labels

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)

# **************SVM Model******************* final.txt
svm_clf_final = SVC(kernel='linear', C=0.1)
svm_clf_final.fit(X_train, y_train)

svm_y_pred = svm_clf_final.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_y_pred)
svm_cm = confusion_matrix(y_test, svm_y_pred)

print("*****************SVM Model - Final*****************")
print(f'Accuracy: {svm_accuracy}')
print('Confusion Matrix:')
print(svm_cm)

svm_scores = cross_val_score(svm_clf_final, X_final, y_final, cv=10)
print(f'Cross-Validation Accuracy: {svm_scores.mean()}\n')

# ********************MLP Model**************************** final.txt
mlp_clf_final = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)

# Train the model
mlp_clf_final.fit(X_train, y_train)

mlp_y_pred = mlp_clf_final.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_y_pred)
mlp_cm = confusion_matrix(y_test, mlp_y_pred)

print("*****************MLP Model - Final*****************")
print(f'Accuracy: {mlp_accuracy}')
print('Confusion Matrix:')
print(mlp_cm)

mlp_scores = cross_val_score(mlp_clf_final, X_final, y_final, cv=10)
print(f'Cross-Validation Accuracy: {mlp_scores.mean()}\n')

# *****************KNN Model************************* final.txt
# Instantiate the kNN classifier with a specific value of k
svm_clf_final = KNeighborsClassifier(n_neighbors=5, metric='euclidean')  # Adjust the value of n_neighbors and metric as needed
# Standardize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# Split the scaled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_final, test_size=0.2, random_state=42)

# Train the Model
svm_clf_final.fit(X_train, y_train)
y_pred = svm_clf_final.predict(X_test)

print("*****************KNN Model - Final*****************")

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))

# *****************Regression Models*****************

# Load tictac_multi.txt and put inputs and outputs into X and y
A = np.loadtxt('tictac_multi.txt')
X_multi = A[:, :9]   # Input features
y_multi = A[:, 9:]   # Output labels

print(X_multi)
print(y_multi)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# *****************Linear Regression*****************

# Select Linear Regression model
linReg_clf = linear_model.LinearRegression()

# Fit the training data
linReg_clf.fit(X_train, y_train)

# Make a prediction
y_pred = linReg_clf.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("*****************Linear Regression Model*****************")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}\n")

# *****************K-Nearest Neighbors Regression*****************

# Select KNN Regression model
knnr_clf = KNeighborsRegressor(n_neighbors=5)

# Fit training data
knnr_clf.fit(X_train, y_train)

# Make a prediction
y_pred = knnr_clf.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("*****************KNN Regression Model*****************")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}\n")

# *****************MLP Regression Model*****************

# Select MLP Regressor
mlpr_clf = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Fit the model to training data
mlpr_clf.fit(X_train, y_train)

# Make prediction
y_pred = mlpr_clf.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("*****************MLP Regression Model*****************")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}\n")


class TicTacToe:
    def __init__(self, model):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.model = model

    # Reset the board after game over. Board returns to 0 array
    def reset(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'

    # Return true if (row, col) inputs are valid moves
    def is_valid_move(self, row, col):
        return 3 > row >= 0 and 0 <= col < 3 and self.board[row][col] == ' '

    def print_board(self):
        print("  0   1   2")
        for i, row in enumerate(self.board):
            print(f'{i} {" | ".join(row)}')
            if i < 2:
                print(' ' + '-' * 11)

    # Check if move is valid, then input the move into the board
    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            return True
        return False

    # If board is full return true, otherwise false
    def is_board_full(self):
        return all(all(cell != ' ' for cell in row) for row in self.board)

    # Switch to next player
    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                return self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                return self.board[0][i]

        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][0] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]

        return None

    def play_game(self):
        while True:
            print(f"Player {self.current_player}'s turn")
            self.print_board()

            if self.current_player == 'X':
                row = int(input('Enter row (0-2): '))
                col = int(input('Enter column (0-2): '))
                self.make_move(row, col)
            else:
                # Use trained model to predict 0's move
                board_state = np.array(self.board).flatten()
                board_state = np.where(board_state == 'X', 1, board_state)
                board_state = np.where(board_state == 'O', -1, board_state)
                board_state = np.where(board_state == ' ', 0, board_state)
                board_state = board_state.astype(int)
                try:
                    move = int(self.model.predict([board_state])[0])
                # move = int(self.model.predict([board_state])[0])    # TODO: Line doesn't work with regression models
                except:  # we have an array and we want one prediction
                    arr = (self.model.predict([board_state])[0])  # should be an array
                    highestIndex = 0
                    highestVal = 0;
                    visited = set()
                    for i in range(len(arr)):
                        if arr[i] > highestVal and i not in visited:
                            visited.add(i)
                            highestIndex = i
                            highestVal = arr[i]
                    print(highestIndex)
                    move = highestIndex
                # move = self.model.predict( [ board_state[0], board_state[1] ] )
                print(f"Board state: {board_state}")
                print(f"Predicted move: {move}")

                row, col = divmod(move, 3)
                print(f"Row, Col: {row}, {col}")
                if not self.make_move(row, col):
                    print("Unable to move")

            winner = self.check_winner()
            if winner:
                print(f"Player {winner} wins!")
                self.print_board()
                break
            elif self.is_board_full():
                print("It's a draw!")
                self.print_board()
                break
            else:
                self.switch_player()


if __name__ == '__main__':
    while True:
        print("TicTacToe Game\n")
        print("Select Model")
        print("------------")
        print("1. Linear SVM\n2. MLP\n3. KNN\n4. Linear Regression\n5. MLP Regression\n6. KNN Regression\n7. Exit")
        model_input = input("Select Model: ")
        match model_input:
            case "1":
                game = TicTacToe(svm_clf_single)
                game.play_game()
            case "2":
                game = TicTacToe(mlp_clf_single)
                game.play_game()
            case "3":
                game = TicTacToe(knn_clf_single)
                game.play_game()
            case "4":
                game = TicTacToe(linReg_clf)
                game.play_game()
            case "5":
                game = TicTacToe(mlpr_clf)
                game.play_game()
            case "6":
                game = TicTacToe(knnr_clf)
                game.play_game()
            case "7":
                break
