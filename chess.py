import time
import random
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, \
    CuDNNLSTM  # used to stop data from being diluted over time, typical of RNN's
from keras.models import load_model
import keras
import csv
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from colorama import Fore, Back, Style
import copy

realPlayer = False
onlyLegalMoves = True
minimax_tree_depth=3
minimax_tree_subnodes=2
#adjusting the board colors https://pypi.org/project/colorama/
board_color1=Back.LIGHTCYAN_EX
board_color2=Back.LIGHTRED_EX
move_color=Back.MAGENTA
# note, see _encoders.py. a line of code was commented out to suppress warnings
# https://keon.io/deep-q-learning/
# https://www.youtube.com/watch?v=aCEvtRtNO-M
class DNNPlayer(object):
    def __init__(self, memSize, gamma, epsilon, epsilonDecay, epsilonMin, learningRate, player):
        self.gamma = gamma  # decay or discount rate, to calculate the future discounted reward
        self.epsilon = epsilon  # exploration rate, this is the rate in which a player randomly makes a move
        self.epsilonDecay = epsilonDecay  # decreases the exploration rate as the number of games played increases
        self.epsilonMin = epsilonMin  # the player should explore at least this amount
        self.learningRate = learningRate  # how much the NN learns after each exploration
        self.player = player
        self.moveCount = 0
        self.error_count = 0
        # d = deque()
        self.memory = deque(maxlen=500000)
        self.model = ""
        self.model_is_built = False

    def buildModel(self):
        print("building " + self.player + " player model...")
        i = 0

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        x_in_order = []
        y_in_order = []
        train_count = 0
        test_count = 0
        switch = True

        # pop all the data from memory into the training format
        # x variable holds the encoded chess board
        # y varaibale represents corresponding binary move
        while (True):
            try:
                data = self.memory.popleft()
                rand = np.random.random()

                # add to each of the training and testing arrays randomly and uniform
                if (rand <= 0.65):
                    x_train.extend(data[0])
                    y_train.extend(data[1])
                    train_count += 1
                    switch = True
                else:
                    x_test.extend(data[0])
                    y_test.extend(data[1])
                    test_count += 1
                    switch = False
                data = ""

            except(IndexError):
                break

        x_train = np.resize(x_train, (-1, 4, 64))
        x_test = np.resize(x_test, (-1, 4, 64))

        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        print("Fitting data...")
        y_train = np.resize(y_train, (x_train.shape[0], 16))  # reshape into arrays of 16
        y_test = np.resize(y_test, (x_test.shape[0], 16))  # reshape into arrays of 16


        print("Shape of data:")
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)

        print("Building model...")
        model = Sequential()
        input_shape = x_train[0].shape
        model.add(LSTM(256, input_shape=(input_shape), activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(256, activation='relu'))
        model.add(Dropout(0.2))

        # add two dense layers
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        # 16 is the shape of the output data, binary representation of a move
        model.add(Dense(16, activation='sigmoid'))

        # custom optimizer
        opt = keras.optimizers.Adam(lr=0.003, decay=1e-3)

        model.compile(optimizer=opt, metrics=['accuracy'], loss='binary_crossentropy')

        # reshape the y_train (1,4,4) input into Dense 16 input array
        y_train = np.reshape(y_train, (-1, 16))  # reshape into arrays of 16

        y_test = np.reshape(y_test, (-1, 16))  # reshape into arrays of 16
        # fit the data and train the model
        model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test))
        # self.model=model.save_weights()
        self.model = model
        self.model_is_built = True
        print("saving model to disk")
        if (self.player == "White"):
            self.model.save('White_AI_Model.h5', True)
        else:
            self.model.save('Black_AI_Model.h5', True)


    # if Neural Network has a legal prediction, make that move.
    # if epsilon val < rand make random move.
    # If NN move is not legal,  or if epsilon val > rand make make minimax move .
    def AImove(self, boardState):
        # give opportunity for random move first
        print("------MAKING A PREDICTION FOR " + self.player + " +-------")
        temp2 = copy.deepcopy(boardState)
        temp2 = convert_to_binary_matrix(temp2, True)
        temp2 = np.resize(temp2, (1, 4, 64))
        temp2 = np.array(temp2)

        #prediction is a 2d array where prediction[][i] = the suggested move
        prediction = (self.model.predict(temp2))
        prediction = prediction.tolist()

        intStr = ""
        # convert the NN output to a usable output
        for i in range(16):
            if ((prediction[0][i]) >= 0.01):
                prediction[0][i] = 1
            else:
                prediction[0][i] = 0

        # convert the binary vector to a move list
        move = []
        s = ''
        # if Neural Network has a legal prediction, make that move.
        for j in range(16):
            if (j % 4 != 0):
                s += (str(prediction[0][j]))
            if (j % 4 == 0 and j != 0 or j == 15):
                move.append(int(s, 2))
                s = ""
        try:
            #try the NN move
            if (boardState[move[1]][move[0]].move(boardState, move[2], move[3])):
                print("Neural Network says:" + encoder(boardState, move))
                return move
            # If NN move is not legal, make minimax move.
            else:
                self.error_count += 1
                if(random.random()<self.epsilon or self.moveCount<=3):
                    list_of_moves = getAvailableMoves(boardState, self.player)
                    move=random.choice(list_of_moves)
                    print("Random move:" + encoder(boardState, move))
                    return move

                mini: minimaxTree = minimaxTree(boardState, self.player, minimax_tree_depth, minimax_tree_subnodes)
                miniMaxMove = mini.get_best_move()
                move = miniMaxMove
                print("MiniMax says:" + encoder(boardState, move))
                return move

        except(AttributeError):
            self.error_count += 1

            mini: minimaxTree = minimaxTree(boardState, self.player, minimax_tree_depth, minimax_tree_subnodes)
            miniMaxMove = mini.get_best_move()
            move = miniMaxMove
            print("MiniMax says:" + encoder(boardState, move))
            return move

           # move = random.choice(getAvailableMoves(boardState, self.player))
            #print("Everything else failed..returning rand move:" + encoder(boardState, move))
            #return move

#append winning game info to train with later
    def remember(self, state, action, next_state, reward, gameComplete, winningPlayer):
        from keras.models import load_model

        action = convert_to_binary_matrix(action, False)
        state = convert_to_binary_matrix(state, True)

        if (gameComplete and winningPlayer == self.player):
            self.memory.append((state, action))
            print("put game info into memory")

        # get the list of moves


# node is an object that holds the associated value a move in a minimax tree
class tree_node(object):
    def __init__(self, state, move, value, player):
        self.node_value = value
        self.move = move
        self.state = state
        self.player = player
        self.children=[]


class minimaxTree(object):

    def __init__(self,  state, player, depth, num_subtrees):
        self.state = state
        self.player = player
        self.depth = depth
        self.tree = []
        self.num_subtrees=num_subtrees
        self.construct_tree(self.state, self.player)
    #finds the best move in ML
    def getEval(self, board, player, ML):  # heuristic
        move_info=[]
        best_move = []
        best_move_val = -99999
        current_move_val = 0
        if (player == "White"):
            enemyPlayer = "Black"
        else:
            enemyPlayer = "White"
        best = -9999999
        numStackItems = 0
        stack = []
        #while there are legal moves available
        while (ML):
            pSum = 0
            eSum = 0
            move = ML.pop()
            #simple heuristic to add up the piece values
            for i in range(8):
                for j in range(8):
                    if board[i][j] != " ":
                        if board[i][j].player == player:
                            pSum += board[i][j].value
                        # the spot in the board contains an enemy piece
                        else:
                            eSum -= board[i][j].value
            current_move_val=pSum+eSum
            if(current_move_val >= best_move_val):
                move_info.clear()
                best_move=move
                best_move_val=current_move_val
                current_move_val=0
                move_info.append(best_move)
                move_info.append(best_move_val)
        return move_info
    #make the minimax tree based on the depth and number of subtrees
    def construct_tree(self,  state, player):
        root =state
        self.tree.append(tree_node( state, None, 0, player))
        iterator=1
        if self.depth <= 1:
            print("You must make a larger tree")
            return False
        list_of_moves = []
        move=[]
        move_val=0


        for level in range (1, self.depth+1):
            list_of_moves = getAvailableMoves(state, player)
            #the height of any tree is = (num_subtrees^h)+1
            for each_node in range (1, (self.num_subtrees**level)+1):
                #get the best move and value associated with it from the move list
                move_info = self.getEval(state, player, copy.deepcopy(list_of_moves))
                try:
                    move = move_info[0]
                    move_val = move_info[1]
                    if (len(list_of_moves) > 1):
                        list_of_moves.remove(move)
                except: IndexError
                #remove the best move

                move_val=move_info[1]
                temp_state=makeMove(copy.deepcopy(state),move[0], move[1], move[2], move[3])
                #store the info into a tree node
                self.tree.append(tree_node(temp_state, move, move_val, player))
                self.tree[iterator-1].children.append(self.tree[-1])
                # switch the evaluating player
                if(each_node%self.num_subtrees==0):
                    state = self.tree[iterator].state
                    iterator += 1

            if (self.tree[iterator].player == "Black"):
                player = "White"
            elif (self.tree[iterator].player == "White"):
                player = "Black"


        return True
    #traverse the tree...find the best move by applying a summation to each initial subtree
    def bfs(self,sub_node):
        height=0
        sum=0
        while(height<self.depth-1):
            for subtrees in range (0, (self.num_subtrees)):
                a=sub_node.children[subtrees]
                sum+=a.node_value
            sub_node=sub_node.children[height]
            height += 1
        return sum
    def get_best_move(self):
        best_val=-9999
        temp_val=0
        index=0
        for subtrees in range (1, (self.num_subtrees+1)):
            subtree_node=self.tree[subtrees]
            temp=self.bfs(subtree_node)
            if(temp>best_val):
                best_val=temp
                index=subtrees
                temp=0
            return self.tree[index].move



        return



    def getMove(self):
        left = self.traverse_tree(self.root.left_child, 0, self.player)
        right = self.traverse_tree(self.root.right_child, 0, self.player)

        if (left >= right):
            return self.root.left_child.move
        else:
            return self.root.right_child.move


def convert_to_binary_matrix(to_encode, is_game_board):
    int_str = ""
    move_to_bin = []
    # to encode the board
    if is_game_board:
        for i in range(8):
            for j in range(8):
                if (to_encode[i][j] != " "):
                    # convert the integer value of each piece to 4 bit binary

                    int_str += (("{0:{fill}4b}".format(to_encode[i][j].getEncodedVal(), fill='0')))
                else:
                    # spot is empty, give value zero
                    int_str += (("{0:{fill}4b}".format(0, fill='0')))
    else:
        for i in range(len(to_encode)):
            int_str += (("{0:{fill}4b}".format(to_encode[i], fill='0')))
    for i in range(len(int_str)):
        move_to_bin.append(int(int_str[i]))
    # print(move_to_bin)
    if (is_game_board):
        move_to_bin = np.reshape(move_to_bin, (4, 64))  # 4 bits*8 columns*8 rows
    else:
        move_to_bin = np.reshape(move_to_bin, (-1, 16))  # 4 bits, 4 integers
    return move_to_bin


def convertData():
    df = pd.read_csv('AI_data.csv', usecols=['Winner', 'ML'])
    print(df)
    df.head()
    # print(df.head(2))
    l = []
    for i in range(160):
        l = df.iloc[i].tolist()
        print(df.iloc[i])

    print(l)


# parent class of all chess piece child objects
class piece(object):
    def __init__(self, player):
        self.player = player
        self.moveCount = 0
        # self.encodedVal = 0

    def getEncodedVal(self):
        return self.encodedVal

    def moveCounter(self):
        self.moveCount += 1

    def getMoveCount(self):
        return self.moveCount

    def updatePos(self, x, y):
        self.posX = x
        self.posY = y

    def toStr(self):
        return self.string

    def isPiece(self):
        return True

    def getPlayer(self):
        return self.player

    def getType(self):
        return self.type

    def getX(self):
        return self.posX

    def getY(self):
        return self.posY

    def setX(self, x):
        self.posX = x

    def setY(self, y):
        self.posY = y

    def setMoveCount(self, mvcnt):
        self.moveCount = mvcnt

    def setGameDate(self, data):
        self.data = data


# used as a helper function to ensure player is not putting their own king in check
def putOwnKingInCheck(board, xPos, yPos, moveX, moveY):
    try:
        # make the mock move
        temp = copy.deepcopy(board)
        player = board[yPos][xPos].getPlayer()
        temp[moveY][moveX] = temp[yPos][xPos]
        temp[yPos][xPos] = " "
        # see if the mock move will put the moving player into check

        chkInfo = inCheck(temp, player)

        if (chkInfo[0] == 1):
            return True
        else:
            return False
    except(AttributeError):
        return True


class pawn(piece):
    def __init__(self, player):
        self.player = player

        self.type = "p"
        self.moveCount = 0
        self.string = " "
        self.value = 1
        self.twoMove = False
        if (player == "Black"):
            self.string = "♙"
            self.encodedVal = 1

        else:
            self.string = "♟"
            self.encodedVal = 2

    def move(self, board, moveX, moveY):
        if (moveY == self.getY() and moveX == self.getX()):
            return False
        # setting up the booleans
        horizontalMovement = False
        verticalMovement = False
        pieceInMoveSpot = False
        hasMoved = False
        pieceInMoveSpot = False
        attackingFriendlyPiece = False
        attackingEnemyPiece = False
        twoSpaceMove = False
        notValidMove = False
        movingBackwards = False

        if (self.getX() != moveX):
            horizontalMovement = True
            # print("horizontalMovement")
        if (self.getY() != moveY):
            verticalMovement = True
        if (self.player == "Black") and self.getY() > moveY:
            movingBackwards = True
            return False
        elif (self.player == "White") and self.getY() < moveY:
            movingBackwards = True
            return False
        if (not horizontalMovement and not verticalMovement):
            return False

            # print("verticalMovement")
        if moveX > 7 or moveX < 0 or moveX > 7 or moveY < 0:
            notValidMove = True
            return False
        if abs(moveY - self.getY()) > 2:
            return False
        if abs(moveY - self.getY()) == 2:
            twoSpaceMove = True
        if abs(moveY - self.getY()) > 1 and abs(moveX - self.getX()) > 0:
            return False
        if (twoSpaceMove and self.getMoveCount() > 1):
            return False
        if (twoSpaceMove and self.player == "White" and board[self.getY() - 1][self.getX()] != " "):
            return False
        if (twoSpaceMove and self.player == "Black" and board[self.getY() + 1][self.getX()] != " "):
            return False

        if (board[moveY][moveX] != " "):
            pieceInMoveSpot = True
            # print("pieceInMoveSpot")
        if (pieceInMoveSpot):
            if (board[moveY][moveX].getPlayer() == self.getPlayer()):
                attackingFriendlyPiece = True
                return False
                # print("attackingFriendlyPiece")
            if (board[moveY][moveX].getType() == "K"):
                return False
            else:
                attackingFriendlyPiece = False
                attackingEnemyPiece = True
        if (attackingEnemyPiece and not verticalMovement):
            return False
            # print("attackingEnemyPiece")
        if not pieceInMoveSpot and horizontalMovement and verticalMovement \
                and not putOwnKingInCheck(board, self.getX(), self.getY(), moveX, moveY):
            if (self.getPlayer() == "Black"):
                if (moveY == 5 and abs(moveY - self.getY() < 2)):
                    if (board[moveY - 1][moveX] != " "):
                        if (board[moveY - 1][moveX].getMoveCount() == 1 and board[moveY - 1][moveX].getType() == "p"):
                            return enPassant(board, self.getX(), self.getY(), moveX, moveY)
            else:
                if (moveY == 2 and abs(moveY - self.getY() < 2)):
                    if (board[moveY + 1][moveX] != " "):
                        if (board[moveY + 1][moveX].getMoveCount() == 1 and board[moveY + 1][moveX].getType() == "p"):
                            return enPassant(board, self.getX(), self.getY(), moveX, moveY)

        if (attackingEnemyPiece):
            if (abs(moveY - self.getY()) > 1) or (
                    abs(moveX - self.getX())) > 1 or not horizontalMovement or twoSpaceMove \
                    or putOwnKingInCheck(board, self.getX(), self.getY(), moveX, moveY):
                return False
            else:
                return True

        if (self.moveCount > 0):
            hasMoved = True
        # print("hasMoved")
        if (abs(moveY - self.getY() > 1)):
            twoSpaceMove = True
        if (abs(moveY - self.getY() > 2)):
            notValidMove = True
        if (abs(moveX - self.getX() > 2)):
            notValidMove = True
        if (attackingEnemyPiece):
            return self.attackMove(board, moveX, moveY)
        if (not hasMoved and twoSpaceMove):
            self.twoMove = True

        if (
                hasMoved and twoSpaceMove
                or movingBackwards
                or pieceInMoveSpot and attackingFriendlyPiece
                or horizontalMovement and not attackingEnemyPiece
                or attackingEnemyPiece and not horizontalMovement
                or notValidMove
                or putOwnKingInCheck(board, self.getX(), self.getY(), moveX, moveY)
        ):
            return False
        else:

            return True

    def attackMove(self, board, moveX, moveY):
        # setting up the booleans
        horizontalMovement = False
        verticalMovement = False
        pieceInMoveSpot = False
        hasMoved = False
        pieceInMoveSpot = False
        attackingFriendlyPiece = False
        attackingEnemyPiece = False
        twoSpaceMove = False
        notValidMove = False
        movingBackwards = False
        if (self.getX() != moveX):
            horizontalMovement = True
            # print("horizontalMovement")
        if (self.getY() != moveY):
            verticalMovement = True
        if (self.player == "Black") and self.getY() > moveY:
            movingBackwards = True
        elif (self.player == "White") and self.getY() < moveY:
            movingBackwards = True

            # print("verticalMovement")
        if moveX > 7 or moveX < 0 or moveX > 7 or moveY < 0:
            notValidMove = True
            return False
        if abs(moveY - self.getY()) > 2 or abs(self.getY() - moveY) > 2:
            return False
        if abs(moveY - self.getY()) == 2 or abs(self.getY() - moveY) == 2:
            twoSpaceMove = True

        if (board[moveY][moveX] != " "):
            pieceInMoveSpot = True
            # print("pieceInMoveSpot")
        if (pieceInMoveSpot):
            if (board[moveY][moveX].getPlayer() == self.getPlayer()):
                attackingFriendlyPiece = True
                # print("attackingFriendlyPiece")
            else:
                attackingFriendlyPiece = False
                attackingEnemyPiece = True
                # print("attackingEnemyPiece")
        if (attackingEnemyPiece):
            if (abs(moveY - self.getY()) > 1) or (
                    abs(moveX - self.getX())) > 1 or not horizontalMovement or twoSpaceMove:
                return False
            else:
                return True
        if (self.moveCount > 0):
            hasMoved = True
        # print("hasMoved")
        if (abs(moveY - self.getY() > 1)):
            twoSpaceMove = True
        if (abs(moveY - self.getY() > 2)):
            notValidMove = True
        if (abs(moveX - self.getX() > 2)):
            notValidMove = True
        if (attackingEnemyPiece):
            return self.attackMove(board, moveX, moveY)

        if (
                twoSpaceMove
                or verticalMovement
                or movingBackwards
                or pieceInMoveSpot and attackingFriendlyPiece
                or horizontalMovement and not attackingEnemyPiece
                or attackingEnemyPiece and not horizontalMovement
                or notValidMove
        ):
            return False
        else:
            return True

    def piecesInWay(self, moveVector, board):
        x = self.posX
        y = self.posY
        while (moveVector.getMag() != 0):
            if (board[moveVector.getX()][moveVector.getY()] != None):
                return False
            if (moveVector.getX() > 0):
                moveVector.minX()
            if (moveVector.getY() > 0):
                moveVector.minY()
        return True
    # todo finish move method for rook


class rook(piece):
    def __init__(self, player):
        self.player = player

        self.type = "r"
        self.moveCount = 0
        self.string = " "
        self.value = 5
        if (player == "Black"):
            self.string = "♖"
            self.encodedVal = 3

        else:
            self.string = "♜"
            self.encodedVal = 4

    def move(self, board, moveX, moveY):
        if (moveY == self.getY() and moveX == self.getX()):
            return False
        horizontalMovement = False
        verticalMovement = False
        pieceInMoveSpot = False
        attackingFriendlyPiece = False
        attackingEnemyPiece = False
        attackingKing = False
        piecesInWay = False
        notValidMove = False
        hasMoved = False
        if moveX > 7 or moveX < 0 or moveY > 7 or moveY < 0:
            notValidMove = True
            return False

        if (self.getX() != moveX):
            horizontalMovement = True
            # print("horizontalMovement")
        if (self.getY() != moveY):
            verticalMovement = True
        if (self.getMoveCount() > 0):
            hasMoved = True
            # print("verticalMovement")

        if (board[moveY][moveX] != " "):
            pieceInMoveSpot = True
        if (pieceInMoveSpot):
            if (board[moveY][moveX].getPlayer() == self.getPlayer()):
                attackingFriendlyPiece = True
                # print("attackingFriendlyPiece")
            else:
                attackingFriendlyPiece = False
                attackingEnemyPiece = True
                # print("attackingEnemyPiece")
        if ((attackingEnemyPiece) and (board[moveY][moveX].getType() == "K")):
            attackingKing = True
        # check each spot]
        if (horizontalMovement):
            mvY = self.getY()
            mvX = self.getX()
            if (moveX > mvX):
                try:
                    while ((mvX != moveX)):
                        mvX += 1
                        if (board[mvY][mvX] != " "):
                            if ((mvX == moveX)):
                                break
                            piecesInWay = True


                except(IndexError, AttributeError):
                    piecesInWay = True

            elif (moveX < mvX):
                try:
                    while ((mvX != moveX)):
                        mvX -= 1
                        if (board[mvY][mvX] != " "):
                            if (mvX == moveX):
                                break
                            piecesInWay = True


                except(IndexError, AttributeError):
                    piecesInWay = True
        if (verticalMovement):
            mvY = self.getY()
            mvX = self.getX()
            if (moveY > mvY):
                try:
                    while ((mvY != moveY)):
                        mvY += 1
                        if (board[mvY][mvX] != " "):
                            if ((mvY == moveY)):
                                break
                            piecesInWay = True


                except(IndexError, AttributeError):
                    piecesInWay = True

            elif (moveY < mvY):
                try:
                    while ((mvY != moveY)):
                        mvY -= 1
                        if (board[mvY][mvX] != " "):
                            if ((mvY == moveY)):
                                break
                            piecesInWay = True


                except(IndexError, AttributeError):
                    piecesInWay = True
        # is the player trying to castle
        # if(attackingKing and not hasMoved and not piecesInWay and attackingFriendlyPiece):
        # return True
        if (
                verticalMovement and horizontalMovement
                or attackingFriendlyPiece
                or attackingKing
                or piecesInWay
                or notValidMove
                or putOwnKingInCheck(board, self.getX(), self.getY(), moveX, moveY)
        ):
            return False
        else:
            return True


class knight(piece):
    def __init__(self, player):
        self.player = player
        self.type = "k"
        self.moveCount = 0
        self.string = " "
        self.value = 3
        self.encodedVal = 4
        if (player == "Black"):
            self.string = "♘"
            self.encodedVal = 5
        else:
            self.string = "♞"
            self.encodedVal = 6

    def move(self, board, moveX, moveY):
        if (moveY == self.getY() and moveX == self.getX()):
            return False
        vertical = abs(abs(moveY) - abs(self.getY()))
        horizontal = abs(abs(moveX) - abs(self.getX()))
        notValidMove = False
        pieceMoveInSpot = False
        attackingFriendlyPiece = False
        if moveX > 7 or moveX < 0 or moveY > 7 or moveY < 0:
            notValidMove = True
        if (notValidMove):
            return False
        if (board[moveY][moveX] != " "):
            pieceMoveInSpot = True
        if (pieceMoveInSpot and board[moveY][moveX].getPlayer() == self.getPlayer()):
            return False

        if (
                horizontal == 2 and vertical == 1
                or vertical == 2 and horizontal == 1
        ):
            return not putOwnKingInCheck(board, self.getX(), self.getY(), moveX, moveY)
            # return True
        else:
            return False


class bishop(piece):
    def __init__(self, player):
        self.player = player
        self.type = "B"
        self.moveCount = 0
        self.string = " "
        self.value = 5
        if (player == "Black"):
            self.string = "♗"
            self.encodedVal = 7
        else:
            self.string = "♝"
            self.encodedVal = 8

    def move(self, board, moveX, moveY):
        if (moveY == self.getY() and moveX == self.getX()):
            return False
        horizontalMovement = False
        verticalMovement = False
        pieceInMoveSpot = False
        attackingFriendlyPiece = False
        attackingEnemyPiece = False
        attackingKing = False
        piecesInWay = False
        notValidMove = False
        x = abs(moveY) - abs(self.getY())
        y = abs(moveX) - abs(self.getX())
        ans = abs(x) - abs(y)
        if (ans != 0):
            notValidMove = True
        if moveX > 7 or moveX < 0 or moveY > 7 or moveY < 0:
            notValidMove = True
            return False

        # print(abs(moveY)-abs(self.getY()))
        # print(abs(moveX)-abs(self.getX()))
        # print(abs(moveY) - abs(moveX) )
        # print(ans)
        if (self.getX() != moveX):
            horizontalMovement = True
            # print("horizontalMovement")
        if (self.getY() != moveY):
            verticalMovement = True
            # print("verticalMovement")

        if (board[moveY][moveX] != " "):
            pieceInMoveSpot = True
        if (pieceInMoveSpot):
            if (board[moveY][moveX].getPlayer() == self.getPlayer()):
                attackingFriendlyPiece = True
                # print("attackingFriendlyPiece")
            else:
                attackingFriendlyPiece = False
                attackingEnemyPiece = True
                # print("attackingEnemyPiece")
        if ((attackingEnemyPiece) and (board[moveY][moveX].getType() == "K")):
            attackingKing = True
        if (horizontalMovement and verticalMovement):
            mvY = self.getY()
            mvX = self.getX()
            if (moveY > mvY and moveX > mvX):
                try:
                    while ((mvY != moveY) or (mvX != moveX)):
                        mvY += 1
                        mvX += 1

                        if (board[mvY][mvX] != " "):
                            if ((mvY == moveY) or (mvX == moveX)):
                                break
                            piecesInWay = True


                except(IndexError, AttributeError):
                    piecesInWay = True

            if (moveY > mvY and moveX < mvX):
                try:
                    while ((mvY != moveY) or (mvX != moveX)):
                        mvY += 1
                        mvX -= 1
                        if (board[mvY][mvX] != " "):

                            # print("2")
                            if ((mvY == moveY) or (mvX == moveX)):
                                break
                            piecesInWay = True


                except(IndexError, AttributeError):
                    piecesInWay = True

            if (moveY < mvY and moveX < mvX):
                try:
                    while ((mvY != moveY) or (mvX != moveX)):
                        mvY -= 1
                        mvX -= 1

                        if (not board[mvY][mvX] == " "):

                            # print("4")
                            if ((mvY == moveY) or (mvX == moveX)):
                                break
                            piecesInWay = True



                except(IndexError, AttributeError):
                    piecesInWay = True

            if (moveY < mvY and moveX > mvX):
                try:
                    while ((mvY != moveY) or (mvX != moveX)):
                        mvY -= 1
                        mvX += 1

                        if (board[mvY][mvX] != " "):

                            # print("here5")
                            if ((mvY == moveY) or (mvX == moveX)):
                                break
                            piecesInWay = True


                except(IndexError, AttributeError):
                    piecesInWay = True

        if (
                not (verticalMovement and horizontalMovement)
                or attackingFriendlyPiece
                or attackingKing
                or piecesInWay
                or notValidMove
                or putOwnKingInCheck(board, self.getX(), self.getY(), moveX, moveY)
                # or not abs(moveY) - abs(moveX) == 0

        ):
            return False
        else:
            return True


class queen(piece):
    def __init__(self, player):
        self.player = player

        self.type = "Q"
        self.moveCount = 0
        self.string = " "
        self.value = 10
        if (player == "Black"):
            self.string = "♕"
            self.encodedVal = 9
        else:
            self.string = "♛"
            self.encodedVal = 10

    def move(self, board, moveX, moveY):
        if (moveY == self.getY() and moveX == self.getX()):
            return False
        if (moveY == self.getY() and moveX == self.getX()):
            return False
        horizontalMovement = False
        verticalMovement = False
        pieceInMoveSpot = False
        attackingFriendlyPiece = False
        attackingEnemyPiece = False
        attackingKing = False
        piecesInWay = False
        notValidMove = False

        if moveX > 7 or moveX < 0 or moveY > 7 or moveY < 0:
            notValidMove = True

        if (self.getX() != moveX):
            horizontalMovement = True

        if (self.getY() != moveY):
            verticalMovement = True

        if (board[moveY][moveX] != " "):
            pieceInMoveSpot = True
        if (pieceInMoveSpot):
            if (board[moveY][moveX].getPlayer() == self.getPlayer()):
                attackingFriendlyPiece = True
                return False
            if (board[moveY][moveX].getType() == "K"):
                return False

            else:
                attackingFriendlyPiece = False
                attackingEnemyPiece = True

        if ((attackingEnemyPiece) and (board[moveY][moveX].getType() == "K")):
            attackingKing = True
            return False
        if (horizontalMovement and verticalMovement):
            mvY = self.getY()
            mvX = self.getX()
            if (moveY > mvY and moveX > mvX):
                try:
                    while ((mvY != moveY) or (mvX != moveX)):
                        mvY += 1
                        mvX += 1

                        if (board[mvY][mvX] != " "):
                            if ((mvY == moveY) and (mvX == moveX)):
                                break
                            piecesInWay = True


                except(IndexError, AttributeError):
                    piecesInWay = True

            elif (moveY > mvY and moveX < mvX):
                try:
                    while ((mvY != moveY) or (mvX != moveX)):
                        mvY += 1
                        mvX -= 1
                        if (board[mvY][mvX] != " "):

                            if ((mvY == moveY) and (mvX == moveX)):
                                break
                            piecesInWay = True


                except(IndexError, AttributeError):
                    piecesInWay = True

            elif (moveY < mvY and moveX < mvX):
                try:
                    while ((mvY != moveY) or (mvX != moveX)):
                        mvY -= 1
                        mvX -= 1

                        if (not board[mvY][mvX] == " "):

                            if ((mvY == moveY) and (mvX == moveX)):
                                break
                            piecesInWay = True



                except(IndexError, AttributeError):
                    piecesInWay = True

            elif (moveY < mvY and moveX > mvX):
                try:
                    while ((mvY != moveY) or (mvX != moveX)):
                        mvY -= 1
                        mvX += 1

                        if (board[mvY][mvX] != " "):

                            if ((mvY == moveY) and (mvX == moveX)):
                                break
                            piecesInWay = True


                except(IndexError, AttributeError):
                    piecesInWay = True
        elif (horizontalMovement):
            mvY = self.getY()
            mvX = self.getX()
            if (moveX > mvX):
                try:
                    while ((mvX != moveX)):
                        mvX += 1
                        if (board[mvY][mvX] != " "):
                            if ((mvX == moveX)):
                                break
                            piecesInWay = True


                except(IndexError, AttributeError):
                    piecesInWay = True

            elif (moveX < mvX):
                try:
                    while ((mvX != moveX)):
                        mvX -= 1
                        if (board[mvY][mvX] != " "):
                            if ((mvX == moveX)):
                                break
                            piecesInWay = True


                except(IndexError, AttributeError):
                    piecesInWay = True
        elif (verticalMovement):
            mvY = self.getY()
            mvX = self.getX()
            if (moveY > mvY):
                try:
                    while ((mvY != moveY)):
                        mvY += 1
                        if (board[mvY][mvX] != " "):
                            if ((mvY == moveY)):
                                break
                            piecesInWay = True


                except(IndexError, AttributeError):
                    piecesInWay = True

            elif (moveY < mvY):
                try:
                    while ((mvY != moveY)):
                        mvY -= 1
                        if (board[mvY][mvX] != " "):
                            if ((mvY == moveY)):
                                break
                            piecesInWay = True


                except(IndexError, AttributeError):
                    piecesInWay = True

        if (
                attackingFriendlyPiece
                or attackingKing
                or piecesInWay
                or notValidMove
                or putOwnKingInCheck(board, self.getX(), self.getY(), moveX, moveY)
                # or not abs(moveY) - abs(moveX) == 0
        ):
            return False
        else:
            return True


class king(piece):
    def __init__(self, player):

        self.player = player

        self.type = "K"
        self.moveCount = 0
        self.value = 10000

        if (self.player == "Black"):
            isBlackKing = True
            self.string = "♔"
            self.encodedVal = 11
        else:
            self.string = "♚"
            self.encodedVal = 12

    def move(self, board, moveX, moveY):
        if (moveY == self.getY() and moveX == self.getX()):
            return False
        castling = False
        moveGreaterThanOne = False
        if (abs(self.getX() - moveX) > 1
                or abs(self.getY() - moveY) > 1
                or moveX < 0 or moveX > 7
                or moveY < 0 or moveY > 7

        ):
            moveGreaterThanOne = True
        if (moveX < 0 or moveX > 7
                or moveY < 0 or moveY > 7
        ):
            return False

        if (board[moveY][moveX] != " "):
            if (board[moveY][moveX].getType() == "K"):
                return False
            # if player is trying to capture his own piece
            if (board[moveY][moveX].getPlayer() == self.getPlayer()) and board[moveY][moveX].getType() != "r":
                return False
            # player is trying to castle
            elif (board[moveY][moveX].getPlayer() == self.getPlayer()) and board[moveY][moveX].getType() == "r":
                castling = True
        if (moveGreaterThanOne and not castling):
            return False
        tempBoard = copy.deepcopy(board)
        tempBoard = makeMove(tempBoard, self.getX(), self.getY(), moveX, moveY)
        inCheckInfo = inCheck(tempBoard, self.getPlayer())

        if (inCheckInfo[0] == 1):
            return False
        # player is trying to castle, is it a valid move?
        '''
                       When are you not allowed to castle?
                       (1)Your king has been moved earlier in the game.
                       (2)The rook that you would castle with has been moved earlier in the game.
                       (3)There are pieces standing between your king and rook.
                       (4)The king is in check.
                       (5)The king moves through a square that is attacked by a piece of the opponent.
                       https://www.chessvariants.com/d.chess/castlefaq.html
                       '''
        if (castling):

            # (1) and (2)✓
            if (board[moveY][moveX].getMoveCount() > 0 or board[self.getY()][self.getX()].getMoveCount() > 0):
                return False
            # (4)
            test = inCheck(board, self.getPlayer());
            # printBoard(board)
            if (test[0] == 1):
                return False
            # (5)

            if (abs(self.getX() - moveX) == 3):

                # (3)(5)✓
                if (board[moveY][moveX - 1] != " "
                        or board[moveY][moveX - 2] != " "
                        or canEnemyMove(board, self.getPlayer(), moveX - 1, moveY)
                        or canEnemyMove(board, self.getPlayer(), moveX - 2, moveY)

                ):
                    return False
                else:

                    return True
            if (abs(self.getX() - moveX) == 4):
                # print("444")
                if (board[moveY][moveX + 1] != " "
                        or board[moveY][moveX + 2] != " "
                        or board[moveY][moveX + 3] != " "
                        or canEnemyMove(board, self.getPlayer(), moveX + 1, moveY)
                        or canEnemyMove(board, self.getPlayer(), moveX + 2, moveY)
                        or canEnemyMove(board, self.getPlayer(), moveX + 3, moveY)

                ):
                    return False
                else:

                    return True

        return True


def printBoard(b):
    switch = True
    incl = False
    sideNum = 8
    bottom = " A " + u'\u2002' + " B " + u'\u2002' + " C " + u'\u2002' + " D " + u'\u2002' + " E " + u'\u2002' + " F " + u'\u2002' + " G " + u'\u2002' + " H "
    printstr = " "
    # rows
    for i in range(8):
        # cols
        for j in range(8):
            if (j == 7):
                incl = True

            if ((b[i][j] != " ")):
                if (switch):
                    printstr = (board_color1 + " " + b[i][j].string + " ")
                else:
                    printstr = (board_color2 + " " + b[i][j].string + " ")
            else:
                if (switch):
                    printstr = (board_color1 + " " + u'\u3000' + " ")
                else:
                    printstr = (board_color2 + " " + u'\u3000' + " ")

            print(printstr, end="")
            switch = not switch
            sys.stdout.write(Style.RESET_ALL)
        print(i + 1)
        sideNum -= 1
        switch = not switch
    print(bottom)

#shows available moves for a selected piece
def printBoardWithAvailMoves(b, x, y, dispBoard):
    if (b[y][x] == " "):
        print("You did not select a Piece!")
        return
    switch = True
    incl = False
    canMove = False
    sideNum = 8
    bottom = " A " + u'\u2002' + " B " + u'\u2002' + " C " + u'\u2002' + " D " + u'\u2002' + " E " + u'\u2002' + " F " + u'\u2002' + " G " + u'\u2002' + " H "
    printstr = " "
    # rows
    for i in range(8):
        # cols
        for j in range(8):
            if (j == 7):
                incl = True

            if (dispBoard and b[y][x].move(b, j, i)):
                canMove = True

            if ((b[i][j] != " ")):
                if (switch):
                    printstr = (board_color1 + " " + b[i][j].string + " ")
                else:
                    printstr = (board_color2+ " " + b[i][j].string + " ")

                if (canMove):
                    printstr = (move_color + " " + b[i][j].string + " ")
            else:
                if (switch):
                    printstr = (board_color1 + " " + u'\u3000' + " ")
                else:
                    printstr = (board_color2 + " " + u'\u3000' + " ")
                if (canMove):
                    printstr = (move_color + " " + u'\u3000' + " ")
            switch = not switch
            print(printstr, end="")
            canMove = False

            sys.stdout.write(Style.RESET_ALL)
        print(i + 1)
        sideNum -= 1
        switch = not switch
    print(bottom)


#set up a new board with piece objects
def newGame():  #
    # initialize the pieces
    pb0: pawn = pawn("Black")
    pb1: pawn = pawn("Black")
    pb2: pawn = pawn("Black")
    pb3: pawn = pawn("Black")
    pb4: pawn = pawn("Black")
    pb5: pawn = pawn("Black")
    pb6: pawn = pawn("Black")
    pb7: pawn = pawn("Black")
    pw0: pawn = pawn("White")
    pw1: pawn = pawn("White")
    pw2: pawn = pawn("White")
    pw3: pawn = pawn("White")
    pw4: pawn = pawn("White")
    pw5: pawn = pawn("White")
    pw6: pawn = pawn("White")
    pw7: pawn = pawn("White")

    rb0: rook = rook("Black")
    rb1: rook = rook("Black")
    rw0: rook = rook("White")
    rw1: rook = rook("White")

    knb0: knight = knight("Black")
    knb1: knight = knight("Black")
    knw0: knight = knight("White")
    knw1: knight = knight("White")

    bb0: bishop = bishop("Black")
    bb1: bishop = bishop("Black")
    bw0: bishop = bishop("White")
    bw1: bishop = bishop("White")

    qb0: queen = queen("Black")
    kb0: king = king("Black")
    qw0: queen = queen("White")
    kw0: king = king("White")

    # initialize the board
    b = [[rb0, knb0, bb0, qb0, kb0, bb1, knb1, rb1],
         [pb0, pb1, pb2, pb3, pb4, pb5, pb6, pb7],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [pw0, pw1, pw2, pw3, pw4, pw5, pw6, pw7],
         [rw0, knw0, bw0, qw0, kw0, bw1, knw1, rw1]]
    # rows Y
    for i in range(8):
        # cols X
        for j in range(8):
            if (b[i][j] != " "):
                # set the positions
                b[i][j].setX(j)
                b[i][j].setY(i)

                # encode the pieces relative to their string and initial position
                # b[i][j].encodedVal = str.encode(b[i][j].toStr()) + str.encode(str(i)) + str.encode(str(j))

    return b


# sets the xPos and yPos for each piece
def setPositions(b):
    for i in range(8):
        # cols X
        for j in range(8):
            if (b[i][j] != " "):
                b[i][j].setX(j)
                b[i][j].setY(i)


#checks if the pawn can make a legal en passant move
def enPassant(board, xPos, yPos, moveX, moveY):
    if (board[yPos][xPos].getType() != "p"):
        return False
    if (board[yPos][xPos].getPlayer() == "Black"):
        if (moveY == 5 and abs(moveY - board[yPos][xPos].getY() < 2)):
            if (board[moveY - 1][moveX] != " "):
                if (board[moveY - 1][moveX].getMoveCount() == 1 and board[moveY - 1][moveX].getType() == "p"):
                    return True
    else:
        if (moveY == 2 and abs(moveY - board[yPos][xPos].getY() < 2)):
            if (board[moveY + 1][moveX] != " "):
                if (board[moveY + 1][moveX].getMoveCount() == 1 and board[moveY + 1][moveX].getType() == "p"):
                    return True


#function for enPassant move
def makeEnPassant(board, xPos, yPos, moveX, moveY):
    if (board[yPos][xPos].getPlayer() == "Black"):
        if (moveY == 5 and abs(moveY - board[yPos][xPos].getY() < 2)):
            if (board[moveY - 1][moveX] != " "):
                if (board[moveY - 1][moveX].getMoveCount() == 1 and board[moveY - 1][moveX].getType() == "p"):
                    board[moveY][moveX] = board[yPos][xPos]
                    board[moveY][moveX].moveCounter()
                    board[moveY][moveX].updatePos(moveX, moveY)
                    board[yPos][xPos] = " "
                    board[moveY - 1][moveX] = " "
                    return board
    else:
        if (moveY == 2 and abs(moveY - board[yPos][xPos].getY() < 2)):
            if (board[moveY + 1][moveX] != " "):
                if (board[moveY + 1][moveX].getMoveCount() == 1 and board[moveY + 1][moveX].getType() == "p"):
                    board[moveY][moveX] = board[yPos][xPos]
                    board[moveY][moveX].moveCounter()
                    board[moveY][moveX].updatePos(moveX, moveY)
                    board[yPos][xPos] = " "
                    board[moveY + 1][moveX] = " "
                    return board


#function for castle move
def castleMove(board, xPos, yPos, moveX, moveY):
    if (moveX > xPos):
        # move the rook first
        # xPos and yPos are the kings' coordinates
        board[yPos][xPos + 1] = board[moveY][moveX]
        # increment the pieces move count
        board[yPos][xPos + 1].moveCounter()
        # update the pieces coordinates
        board[yPos][xPos + 1].updatePos(xPos + 1, yPos)
        # set the previous location to empty
        board[moveY][moveX] = " "

        # move the king next
        board[yPos][xPos + 2] = board[yPos][xPos]

        board[yPos][xPos + 2].moveCounter()
        board[yPos][xPos + 2].updatePos(xPos + 2, yPos)
        board[yPos][xPos] = " "
        return board
    else:
        # move the rook first
        board[yPos][xPos - 1] = board[moveY][moveX]
        # increment the pieces move count
        board[yPos][xPos - 1].moveCounter()
        # update the pieces coordinates
        board[yPos][xPos - 1].updatePos(xPos - 1, yPos)
        # set the previous location to empty
        board[moveY][moveX] = " "
        # move the king next
        board[yPos][xPos - 2] = board[yPos][xPos]
        board[yPos][xPos - 2].moveCounter()
        board[yPos][xPos - 2].updatePos(xPos - 2, yPos)
        board[yPos][xPos] = " "
        return board


#function for pawn promotion
def pawnPromotion(board, xPos, yPos):
    p = ""

    while (p != "q" or p != "r" or p != "b" or p != "r" or p != "k"):

        '''
        print("Select a type of piece to promote your pawn to:")
        print("q for Queen")
        print("r for Rook")
        print("b for Bishop")
        print("k for Knight")
        '''
        if not realPlayer:
            p = "q"
        else:
            p = input()

        if (p == "q"):
            q1: queen = queen(board[yPos][xPos].getPlayer())
            q1.setMoveCount(board[yPos][xPos].getMoveCount())
            # todo set game data
            board[yPos][xPos] == " "
            board[yPos][xPos] = q1
            board[yPos][xPos].updatePos(xPos, yPos)
            return board
        if (p == "r"):
            r1: rook = rook(board[yPos][xPos].getPlayer())
            r1.setMoveCount(board[yPos][xPos].getMoveCount())
            # todo set game data
            board[yPos][xPos] == " "
            board[yPos][xPos] = r1
            board[yPos][xPos].updatePos(xPos, yPos)
            return board
        if (p == "b"):
            b1: bishop = bishop(board[yPos][xPos].getPlayer())
            b1.setMoveCount(board[yPos][xPos].getMoveCount())
            # todo set game data
            board[yPos][xPos] == " "
            board[yPos][xPos] = b1
            board[yPos][xPos].updatePos(xPos, yPos)
            return board
        if (p == "k"):
            k1: knight = knight(board[yPos][xPos].getPlayer())
            k1.setMoveCount(board[yPos][xPos].getMoveCount())
            # todo set game data
            board[yPos][xPos] == " "
            board[yPos][xPos] = k1
            board[yPos][xPos].updatePos(xPos, yPos)
            return board
    return board


def makeMove(board, xPos, yPos, moveX, moveY):
    castling = False
    if (board[yPos][xPos] != " "):
        if (board[yPos][xPos].type== "p"):
            if (enPassant(board, xPos, yPos, moveX, moveY)):
                return makeEnPassant(board, xPos, yPos, moveX, moveY)

    if (board[moveY][moveX] != " ") and board[yPos][xPos] != " ":
        if (board[yPos][xPos].getType() == "K"
                and board[moveY][moveX].getType() == "r"
                and board[moveY][moveX].getPlayer() == board[yPos][xPos].getPlayer()
        ):
            castling = True

    try:
        if (castling):
            return castleMove(board, xPos, yPos, moveX, moveY)
        else:
            board[moveY][moveX] = board[yPos][xPos]
            board[moveY][moveX].moveCounter()
            board[moveY][moveX].updatePos(moveX, moveY)
            board[yPos][xPos] = " "

            if (board[moveY][moveX].getType() == "p"
                    and board[moveY][moveX].getPlayer() == "White"
                    and moveY == 0
            ):
                print("white pawn promotion")
                return pawnPromotion(board, moveX, moveY)

            if (board[moveY][moveX].getType() == "p"
                    and board[moveY][moveX].getPlayer() == "Black"
                    and moveY == 7
            ):
                print("black pawn promotion")
                return pawnPromotion(board, moveX, moveY)
            return board
    except(AttributeError, IndexError):

        # print("That is not a valid move!")
        return board


#returns a list with both king coords
def findKingPos(board):
    coordinates = [0, 0, 0, 0]
    for i in range(8):
        for j in range(8):
            if (board[i][j] != " "):
                if (board[i][j].getPlayer() == "Black") and (board[i][j].getType() == "K"):
                    coordinates[0] = i
                    coordinates[1] = j
                elif (board[i][j].getPlayer() == "White") and (board[i][j].getType() == "K"):
                    coordinates[2] = i
                    coordinates[3] = j
    return coordinates


# test if @param player is in check, returns the x,y coordinates of the attacking piece
def inCheck(board, player):
    coordinates = findKingPos(board)
    bKingY = coordinates[0]
    bKingX = coordinates[1]
    wkingY = coordinates[2]
    wkingX = coordinates[3]
    # use a piece other than a king to test if black is in check or not
    pb0: pawn = pawn("Black")
    pb1: pawn = pawn("White")
    r = [0, -1, -1]
    kingX = 0
    kingY = 0

    temp = copy.deepcopy(board)
    #get the king coordinates
    if (player == "White"):
        temp[wkingY][wkingX] = pb1
        kingX = wkingX
        kingY = wkingY
    elif (player == "Black"):
        temp[bKingY][bKingX] = pb0
        kingX = bKingX
        kingY = bKingY

    for i in range(8):
        for j in range(8):
            if (temp[i][j] != " "):
                if (temp[i][j].getType() != "K"):
                    if temp[i][j].getPlayer() != player and temp[i][j].move(temp, kingX, kingY):
                        r = [1, i, j]
                        return r
    return r


def checkMate(board, player, coordinates):
    # assume that the player is in checkmate unless they are not.
    '''
    how to get out of a checkmate:
    (1): move the king to a new location where it will not be in check
    (2): capture the attacking piece
    (3): move a friendly piece in such a way that the player is no longer in check
    '''
    kingX, kingY = 0, 0
    kingCoords = findKingPos(board)
    if (player == "Black"):

        kingY = kingCoords[0]
        kingX = kingCoords[1]
        print(kingY)
        print(kingX)
    else:
        kingY = kingCoords[2]
        kingX = kingCoords[3]

    temp = copy.deepcopy(board)

    for i in range(8):
        for j in range(8):
            # (1)
            if (temp[kingY][kingX].move(temp, j, i)):
                # print("Case 1")

                return False

            if (temp[i][j] != " "):
                # (2)

                inChckMate = inCheck(temp, player)
                if (temp[i][j].getPlayer() == player and temp[i][j].move(temp, coordinates[2], coordinates[1])
                        and inChckMate[0] == 0
                ):
                    # print("Case 2")

                    return False
                # (3)
                #  checking every position if it is a friendly piece
                #  and if it can move, make the move.
                # if that move gets the player out of check, then return false
                # else undo the move, check the next available move
                # extremely inefficient but it works for now
                if (temp[i][j].getPlayer() == player):
                    for x in range(8):
                        for y in range(8):
                            if (temp[i][j].move(temp, y, x)):
                                temp = makeMove(temp, temp[i][j].getX(), temp[i][j].getY(), y, x)
                                # printBoard(temp)
                                inChckMate = inCheck(temp, player)
                                if not (inChckMate[0] == 1):
                                    # printBoard(temp)
                                    return False
                                else:
                                    temp = copy.deepcopy(board)

    # print("checkmate")
    return True


# converts a user entered string to board indices
def convertInput(xInput, yInput):
    x = xInput[0].upper()
    x = ord(x)
    x = x - 65 #
    y = int(yInput)
    y -= 1
    r = [x, y]
    return r


# helper function for InCheckMate and king.move() functions
def canEnemyMove(board, player, x, y):
    for i in range(8):
        for j in range(8):
            if (board[i][j] != " "):
                if (board[i][j].getPlayer() != player):
                    if (board[i][j].type != "K"):
                        if (board[i][j].move(board, x,
                                             y)):  # and board[i][j].move(board, x, y) and board[i][j].getType()!="K":
                            print(board[i][j].toStr())
                            print("can move there")
                            return False

    return False


# object placeholder for move Information
class moveInfo():
    def __init__(self):
        self.xPos = None
        self.yPos = None
        self.moveX = None
        self.moveY = None
        self.next = None

    def getX(self):
        return self.xLoc()

    def getY(self):
        return self.yLoc()

    def moveX(self):
        return self.moveX()

    def moveY(self):
        return self.moveY()

    def addMove(self, xPos, yPos, moveX, moveY):
        if (self.xPos == None):
            self.xPos = xPos
            self.yPos = yPos
            self.moveX = moveX
            self.moveY = moveY
            self.next = None
            return
        while (self != None):
            self = self.next

        self.xPos = xPos
        self.yPos = yPos
        self.moveX = moveX
        self.moveY = moveY
        self.next = None

    def getHead(self):
        while (self.prev != None):
            self = self.prev
        return self


# convert figuirine algebraic to board indices.
def decoder(moveStr):
    temp = []
    temp = moveStr.tolist()
    print(temp[0])

    # moveStr=np.array(moveStr)
    # 	moveStr will have a value similair to this: ♘f3♞c6

    # moveStr=array(moveStr)
    xPos = temp[0][1]
    yPos = temp[0][2]
    moveX = temp[0][4]
    moveY = temp[0][5]
    selPce = convertInput(xPos.upper(), yPos)
    moveLoc = convertInput(moveX.upper(), moveY)
    moveInfo = [selPce[0], selPce[1], moveLoc[0], moveLoc[1]]
    return moveInfo


# encode a move to figuirine algebraic
def encoder(board, availableMove):
    encodedStr = ""
    xPos = int(availableMove[0])
    yPos = int(availableMove[1])
    moveX = int(availableMove[2])
    moveY = int(availableMove[3])
    try:
        player = board[yPos][xPos].getPlayer()
    except(AttributeError):
        return False
    if (player == "White"):
        enemyPlayer = "Black"
        inCheckInfo = inCheck(board, enemyPlayer)
    else:
        enemyPlayer = "White"
        inCheckInfo = inCheck(board, enemyPlayer)

    # convert the move info to figurine algebraic
    # 	♘f3♞c6
    # https://en.wikipedia.org/wiki/Chess_notation
    convertedXpos = xPos + 65
    convertedXpos = chr(convertedXpos)
    convertedYpos = yPos + 1

    convertedMoveX = moveX + 65
    convertedMoveX = chr(convertedMoveX)
    convertedMoveY = moveY + 1

    encodedStr += board[yPos][xPos].toStr() + convertedXpos + (str(convertedYpos))
    if (board[moveY][moveX] != " "):
        encodedStr += board[moveY][moveX].toStr()

    else:
        encodedStr += " "
    encodedStr += convertedMoveX
    encodedStr += str(convertedMoveY)

    # see if enemy player is in check or checkMate
    if (inCheckInfo[0] == 1):
        if (checkMate(board, enemyPlayer, inCheckInfo)):
            encodedStr += "#"  # represents a checkmate
        else:
            encodedStr += "+"  # represents a check
    return encodedStr


# return a vector of encoded available moves
def getAvailableMovesEncoded(board, player):
    encodedVector = []
    index = 0
    listOfMoves = getAvailableMoves(board, player)
    while (index < len(listOfMoves)):
        availableMove = listOfMoves.pop(index)
        index += 1
        encodedVector.append(encoder(board, availableMove))
    return encodedVector


# return a list of uncoded availble moves as board indices
def getAvailableMoves(board, player):
    # holds the coordinates of the piece and the location to move
    # find a piece belonging to player

    mainList = []
    l = [0, 0, 0, 0]
    # rows
    for i in range(8):
        # cols
        for j in range(8):
            if board[i][j] != " ":
                if board[i][j].getPlayer() == player:
                    # find all the spots on the board that piece can move
                    # rows
                    for y in range(8):
                        # cols
                        for x in range(8):
                            # if board[i][j] != " ":
                            if board[i][j].move(board, x, y):
                                # get the info of the legal move
                                l = [j, i, x, y]
                                # store the info in the main list
                                mainList.append(l)

    return mainList


# return an evaluation to act as a fitness function to train the DNN
def getEvaluation(board, player, moveCount):
    moveSum = 0
    enemymMoveSum = 0
    pieceTotalVal = 0
    enemyTotalVal = 0
    summation = 0;

    for i in range(8):
        for j in range(8):
            for x in range(8):
                for y in range(8):
                    if (board[i][j] != " "):
                        if (board[i][j].getPlayer() == player):
                            pieceTotalVal += board[i][j].value
                        if (board[i][j].getPlayer() != player):
                            enemyTotalVal += board[i][j].value
                        if (board[i][j].move(board, x, y) and board[i][j].getPlayer() == player):
                            moveSum += board[i][j].value
                        elif (board[i][j].move(board, x, y) and board[i][j].getPlayer() != player):
                            enemymMoveSum += board[i][j].value
    summation = (moveSum + pieceTotalVal - enemyTotalVal - enemymMoveSum)
    summation = summation / ((moveCount + 2) ^ 2)
    inCheckInfo = inCheck(board, "White")
    if (inCheckInfo[0] == 1 and "White" != player):
        summation *= 1000
    inCheckInfo = inCheck(board, "Black")
    if (inCheckInfo[0] == 1 and "Black" != player):
        summation *= 1000

    return (summation)


# two AI game for training and observation
def twoAIGame():
    # def __init__(self, memSize, gamma, epsilon, epsilonDecay, epsilonMin, learningRate, player):

    gamma = 0.95  # decay or discount rate, to calculate the future discounted reward
    epsilon = .95  # exploration rate, this is the rate in which a player randomly makes a move
    epsilonDecay = 0.20  # decreases the exploration rate as the number of games played increases
    epsilonMin = 0.01  # the player should explore at least this amount
    learningRate = 0.01  # how much the NN learns after each exploration
    wp: DNNPlayer = DNNPlayer(50000, gamma, epsilon, epsilonDecay, epsilonMin, learningRate, "White")
    bp: DNNPlayer = DNNPlayer(50000, gamma, epsilon, epsilonDecay, epsilonMin, learningRate, "Black")

    max_num_moves = 45
    numGames = 0
    whiteWins = 0
    blackWins = 0
    whiteInCheck = 0
    blackInCheck = 0
    draw = 0
    recordMoves = []
    counter = 0
    inCheckMate = False
    print("Training....")
    print("Loading saved model for White.... ")
    wp.model = load_model('White_AI_Model.h5')
    print("Loading saved model for Black.... ")
    bp.model = load_model('Black_AI_Model.h5')
    while (numGames < 10000):



        board = newGame()


        numGames += 1
        winner = ""
        # clear the moveList
        recordMoves.clear()

        counter = 0
        # if (wp.model_is_built):
        # print the info every 100 games
        if (numGames % 1000 == 0):
            print("Number of games played: ")
            print(numGames)
            print("Wins for White: ")
            print(whiteWins)
            print("Wins for Black: ")
            print(blackWins)
            print("Draws: ")
            print(draw)
            print("White put into check: ")
            print(whiteInCheck)
            print("Black put into check: ")
            print(blackInCheck)
            print("Black moveError count: ")
            print(bp.error_count)
            print("White moveError count: ")
            print(wp.error_count)


            if (max_num_moves > 25):
                max_num_moves -= 1

            #decrease the exploration rate
            wp.epsilonDecay -= 0.001
            bp.epsilonDecay -= 0.001
            wp.epsilon *= epsilonDecay
            bp.epsilon *= epsilonDecay
            # build and update the models
            wp.buildModel()
            bp.buildModel()
            print("Loading saved model for White.... ")
            wp.model = load_model('White_AI_Model.h5')
            # if (bp.model_is_built):
            print("Loading saved model for Black.... ")
            bp.model = load_model('Black_AI_Model.h5')
            blackInCheck = 0
            whiteInCheck = 0

        inCheckMate = False
        try:
            print("Black error count:")
            print(bp.error_count)
            print("White error count:")
            print(wp.error_count)
            print("mvcnt")
            print(wp.moveCount)
            wp.moveCount = 0
            bp.moveCount = 0
            time.sleep(5)

            while (not inCheckMate):

                # train on a 50 move limit draw
                if (counter > max_num_moves):
                    counter = 0
                    draw += 1
                    winner = "Draw"
                    print(winner + " number of moves")
                    print(max_num_moves)

                    print("clearing memory that resulted in a draw")

                    for i in range(int( len(wp.memory)-1)):  #
                        wp.memory.popleft()
                    for i in range(int( len(bp.memory)-1)):
                        bp.memory.popleft()

                    # go to the outer loop
                    break
                    # reset the epsilon value to avoid learning plateau
                    # wp.epsilon = epsilon
                    # bp.epsilon = epsilon

                counter += 1
                reward = 0
                temp = copy.deepcopy(board)

                try:
                    # try to return a move based on model, if move is not legal, return a random move
                    mi = wp.AImove(temp)
                    print(encoder(temp, mi))
                    board2 = makeMove(temp, mi[0], mi[1], mi[2], mi[3])  # update the board
                    wp.moveCount += 1
                    # printBoard(board2)
                except(RecursionError):
                    break
                # store the previous state, move, current state, reward, and if white is in checkmate
                wp.remember(board, mi, board2, 0, inCheckMate, "")

                # update the board
                board = copy.deepcopy(board2)
                try:
                    inCheckInfo = inCheck(board, "Black")
                except(RecursionError):
                    break

                # see if black is in checkmate
                if (inCheckInfo[0] == 1):
                    if (checkMate(board, "Black", inCheckInfo)):
                        print("Black is in Checkmate!")
                        whiteWins += 1
                        winner = "White"
                        inCheckMate = True
                        printBoard(board)
                        break
                    else:
                        print("Black is in check!")
                        printBoard(board)
                        whiteInCheck += 1
                temp = copy.deepcopy(board)
                try:
                    # try to return a move based on model, if move is not legal, return a random move
                    mi = bp.AImove(temp)
                    print(encoder(temp, mi))  # print the algebraic notation encoded move
                    board2 = makeMove(temp, mi[0], mi[1], mi[2], mi[3])  # update the board
                    bp.moveCount += 1
                    # printBoard(board2)  # print the board
                except(RecursionError):
                    break
                # reward = getEvaluation(board, bp.player, counter)
                # store the previous state, move, current state, reward, and if black is in checkmate
                bp.remember(board, mi, board2, 0, inCheckMate, "")

                board = copy.deepcopy(board2)
                # store the movelist for adding to csv file
                recordMoves.extend(mi)
                # see if white is in check
                try:
                    inCheckInfo = inCheck(board, "White")
                except(RecursionError):
                    break
                # see if white is in checkMate
                if (inCheckInfo[0] == 1):
                    #see if white is in checkmate
                    if (checkMate(board, "White", inCheckInfo)):
                        print("White is in Checkmate!")
                        blackWins += 1
                        winner = "Black"
                        inCheckMate = True
                        printBoard(board)
                        break
                    else:
                        print("White is in Check")
                        printBoard(board)
                        blackInCheck += 1

            # write the data to a .csv file
            if (winner != "Draw" or winner != ""):
                writeDataToExcel(recordMoves, counter, winner, blackInCheck, whiteInCheck, blackWins, whiteWins,
                                 numGames)
                wp.epsilonDecay -= 0.01
                bp.epsilonDecay -= 0.01
                wp.learningRate += 0.01
                bp.learningRate += 0.01
                if (winner == "Black"):
                    bp.remember(board, mi, board, reward, inCheckMate, "Black")
                # bp.buildModel()
                elif (winner == "White"):
                    wp.remember(board, mi, board, reward, inCheckMate, "White")
                    # wp.buildModel()





        except():
            print("-------------------------------------------------------------------there was an error")
            inCheckMate = True

    print(
        "END OF EVALUATION-------------------------------------------------------------------------------------------------------------------------")
    print("Wins for White: ")
    print(whiteWins)
    print("Wins for Black: ")
    print(blackWins)
    print("Draws: ")
    print(draw)
    print("White put into check: ")
    print(whiteInCheck)
    print("Black put into check: ")
    print(blackInCheck)
    #save the model information and rebuild the model
    wp.buildModel()
    bp.buildModel()
    return True


# test efficiency of DNN
def tester():
    wp: DQNNPlayer = DQNNPlayer(50000, 0.95, 1.0, 0.995, 0.01, 0.001, "White")
    # wp=
    wp.buildModel()
    wp.model = load_model("AI_Chess_Model(1).h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    print("evaluating loaded model...")
    # wp=wp.buildModel()

    print("Loaded model from disk")

    # wp=keras.models.load_model("AI_Chess_Model(2).h5")
    print("Loaded model from disk")

    # wp.buildModel()

    numGames = 0
    whiteWins = 0
    blackWins = 0
    whiteInCheck = 0
    blackInCheck = 0
    draw = 0
    recordMoves = []
    counter = 0
    inCheckMate = False
    # board = newGame()
    print("Training....")
    while (numGames < 1000):
        board = newGame()
        numGames += 1
        winner = ""
        # clear the moveList
        recordMoves.clear()
        counter = 0

        # print the info every 100 games
        if (numGames % 100 == 0):
            print("Number of games played: ")
            print(numGames)
            print("Wins for White: ")
            print(whiteWins)
            print("Wins for Black: ")
            print(blackWins)
            print("Draws: ")
            print(draw)
            print("White put into check: ")
            print(whiteInCheck)
            print("Black put into check: ")
            print(blackInCheck)
            blackInCheck = 0
            whiteInCheck = 0
        inCheckMate = False
        try:
            while (not inCheckMate):

                # train on a 50 move limit draw
                if (counter > 49):
                    # print("50 move limit reached! match is a draw!")
                    counter = 0
                    draw += 1
                    winner = "Draw"
                    print(winner)
                    break
                counter += 1

                # return a  random legal move
                mi = wp.AImove(board)
                # make the move
                board2 = makeMove(board, mi[0], mi[1], mi[2], mi[3])  # update the board
                reward = getEvaluation(board, wp.player, counter)

                wp.remember(board, mi, reward, board2, inCheckMate)
                # store the move list for adding to csv file
                recordMoves.extend(mi)
                # see if black is in check
                inCheckInfo = inCheck(board, "Black")
                # see if black is in checkmate
                if (inCheckInfo[0] == 1):
                    if (checkMate(board, "Black", inCheckInfo)):
                        print("Black is in Checkmate!")
                        whiteWins += 1
                        winner = "White"
                        inCheckMate = True
                        printBoard(board)
                        break

                    else:
                        # print("Black is in Check")
                        whiteInCheck += 1
                    # printBoard(board)
                # return a random legal move
                mi = random.choice(getAvailableMoves(board, "Black"))
                # make the move
                board2 = makeMove(board, mi[0], mi[1], mi[2], mi[3])
                # reward = getEvaluation(board, bp.player, counter)
                bp.remember(board, mi, reward, board2, inCheckMate)
                # store the movelist for adding to csv file
                recordMoves.extend(mi)
                # see if white is in check
                inCheckInfo = inCheck(board, "White")
                # see if white is in checkMate
                if (inCheckInfo[0] == 1):
                    if (checkMate(board, "White", inCheckInfo)):
                        print("White is in Checkmate!")
                        blackWins += 1
                        winner = "Black"
                        inCheckMate = True
                        printBoard(board)
                        break
                    else:
                        # print("White is in Check")
                        blackInCheck += 1

            if (winner != "Draw"):
                writeDataToExcel(recordMoves, counter, winner, blackInCheck, whiteInCheck, blackWins, whiteWins,
                                 numGames)





        except(IndexError):
            print("-------------------------------------------------------------------there was an error")
            inCheckMate = True

    print(
        "END OF EVALUATION-------------------------------------------------------------------------------------------------------------------------")
    print("Wins for White: ")
    print(whiteWins)
    print("Wins for Black: ")
    print(blackWins)
    print("Draws: ")
    print(draw)
    print("White put into check: ")
    print(whiteInCheck)
    print("Black put into check: ")
    print(blackInCheck)
    return True


def writeDataToExcel(moveList, moveCount, winner, blackInCheck, whiteInCheck, blackWins, whiteWins, numGames):
    row = [winner, moveCount, moveList, blackInCheck, whiteInCheck, blackWins, whiteWins, numGames]
    with open('AI_data.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()


def two_player_game():
    whiteTurn = True
    inChk = False
    inCheckMate = False
    l = [0, 0, 0]
    board=newGame()
    coordinates = findKingPos(board)
    showAvailMoves = True
    # initilize 2d board with pieces
    board = newGame()
    while (not inCheckMate):
        # inCheckInfo is a list containing whether or not the player is in check
        # and holds the coordinates of the attacking piece
        inCheckInfo = inCheck(board, "White")
        if (inCheckInfo[0] == 1):
            if (checkMate(board, "White", inCheckInfo)):
                print("White is in Checkmate!")
                inCheckMate = True
            else:
                print("White is in Check")
        inCheckInfo = inCheck(board, "Black")
        if (inCheckInfo[0] == 1):
            if (checkMate(board, "Black", inCheckInfo)):
                print("Black is in Checkmate!")
                inCheckMate = True
            else:
                print("Black is in Check")
        try:
            print("Select a Piece")
            p = input()
            yVal = p[1]
            yVal = int(yVal)
            SelPce = convertInput(p[0].upper(), yVal)
            printBoardWithAvailMoves(board, SelPce[0], SelPce[1], showAvailMoves)
            print("Select a move spot")
            p = input()
            yVal = p[1]
            yVal = int(yVal)
            move = convertInput(p[0].upper(), yVal)
            yVal = int(yVal)
            board = makeMove(board, SelPce[0], SelPce[1], move[0], move[1])
        except(ValueError, NameError, AttributeError, IndexError):
            print("Use correct input")
        printBoard(board)
        whiteTurn = not whiteTurn
    print("GAME OVER")


class driver():
    whiteTurn = True
    inChk = False
    inCheckMate = False
    l = [0, 0, 0]
    # initilize 2d board with pieces
    board = newGame()


    #get the coordinates for king positions
    coordinates = findKingPos(board)
    showAvailMoves = True
    print("Is this game an AI game for data collection or training? ")
    print("y for yes, t for train or any other key for no.")

    ai = input()
    test = False
    if (ai == "y"):
        # realPlayer = False
        train = False
        AIPlayer = False
    if (ai == "t"):
        inCheckMate = twoAIGame()
    if (ai == "test"):
        inCheckMate = tester()

    # print out the board with pieces and color
    printBoard(board)
    two_player_game()
