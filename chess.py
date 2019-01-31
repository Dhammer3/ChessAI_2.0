import array

import random
import sys
import pandas as pd


from llist import dllist, dllistnode
import openpyxl
import csv
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for training on gpu


from colorama import Fore, Back, Style
import copy

realPlayer = False
# used for the test cases
onlyLegalMoves = True


# todo fix bug in queen and bishop move logic-----todo DONE ✓
# todo the inCheck or checkmate method DONE✓
# castling the king and rook ----todo DONE✓
# ensure the move methods reflects the above----todo DONE✓
# todo set if statement to determine if player is AI or not
class neuralNetwork():
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)


class piece(object):
    def __init__(self, player):
        self.player = player
        self.moveCount = 0

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

    def moveIsValid(self, moveVector, board, moveX, moveY):
        checkSpot = board[moveY][moveX]
        if (checkSpot != " "):
            boardSpotIsEmpty = False
        else:
            boardSpotIsEmpty = True
        if (not (boardSpotIsEmpty)):
            if (checkSpot.getPlayer() == self.getPlayer()) or (checkSpot.getType == "K"):
                return False
        if (boardSpotIsEmpty and self.getType() == "p"
                and self.moveCount > 0 and abs(moveVector.getY()) != 1):
            return False;
        else:
            return True


def putOwnKingInCheck(board, xPos, yPos, moveX, moveY):
    try:
        temp = copy.deepcopy(board)
        player = board[yPos][xPos].getPlayer()
        temp[moveY][moveX] = temp[yPos][xPos]
        temp[yPos][xPos] = " "
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
        self.value = 1
        self.type = "p"
        self.moveCount = 0
        self.string = " "
        self.twoMove = False
        if (self.player == "Black"):
            self.string = "♙"
        else:
            self.string = "♟"

    def move(self, board, moveX, moveY):
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

        if (board[moveY][moveX] != " "):
            pieceInMoveSpot = True
            # print("pieceInMoveSpot")
        if (pieceInMoveSpot):
            if (board[moveY][moveX].getPlayer() == self.getPlayer()):
                attackingFriendlyPiece = True
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
        self.value = 1
        self.type = "r"
        self.moveCount = 0
        self.string = " "
        if (self.player == "Black"):
            self.string = "♖"
        else:
            self.string = "♜"

    def move(self, board, moveX, moveY):
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
        self.value = 1
        self.type = "k"
        self.moveCount = 0
        self.string = " "
        if (self.player == "Black"):
            self.string = "♘"
        else:
            self.string = "♞"

    def move(self, board, moveX, moveY):
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

                # and attackingFriendlyPiece
        ):
            return not putOwnKingInCheck(board, self.getX(), self.getY(), moveX, moveY)
            # return True
        else:
            return False


class bishop(piece):
    def __init__(self, player):
        self.player = player
        self.value = 1
        self.type = "B"
        self.moveCount = 0
        self.string = " "
        if (self.player == "Black"):
            self.string = "♗"
        else:
            self.string = "♝"

    def move(self, board, moveX, moveY):
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
        self.value = 1
        self.type = "Q"
        self.moveCount = 0
        self.string = " "
        if (self.player == "Black"):
            self.string = "♕"
        else:
            self.string = "♛"

    def move(self, board, moveX, moveY):
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
        self.value = 1
        self.type = "K"
        self.moveCount = 0
        self.string = " "
        if (self.player == "Black"):
            isBlackKing = True
            self.string = "♔"
        else:
            self.string = "♚"

    def move(self, board, moveX, moveY):
        bool = True
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
                bool = False
            # player is trying to castle
            elif (board[moveY][moveX].getPlayer() == self.getPlayer()) and board[moveY][moveX].getType() == "r":
                castling = True
        if (moveGreaterThanOne and not castling):
            return False
        if (putOwnKingInCheck(board, self.getX(), self.getY(), moveX, moveY)):
            return False

        '''
        # will the king be put into check after it makes the move?
        for i in range(8):
            for j in range(8):
                if (board[i][j] != " "):
                    if (board[i][j].getType() != "K"):
                        if ((board[i][j].getType() == "p") and (board[i][j].getPlayer() != self.getPlayer()) and
                                board[i][j].attackMove(board, moveX, moveY)):
                            return False

                        if (board[i][j].getPlayer() != self.getPlayer()) and board[i][j].move(board, moveX, moveY) and \
                                board[i][j].getType() != "p":
                            return False
        '''
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
        if (castling and bool == True):
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

        return bool


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
                    switch = not switch
                    printstr = (Back.LIGHTWHITE_EX + " " + b[i][j].string + " ")
                else:
                    switch = not switch
                    printstr = (Back.LIGHTBLACK_EX + " " + b[i][j].string + " ")
            else:
                if (switch):
                    switch = not switch
                    # print to the same line
                    printstr = (Back.LIGHTWHITE_EX + " " + u'\u2003' + " ")
                else:
                    switch = not switch
                    printstr = (Back.LIGHTBLACK_EX + " " + u'\u2003' + " ")

            print(printstr, end="")

            sys.stdout.write(Style.RESET_ALL)
        print(i + 1)
        sideNum -= 1
        switch = not switch
    print(bottom)


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
                    switch = not switch
                    printstr = (Back.LIGHTWHITE_EX + " " + b[i][j].string + " ")
                else:
                    switch = not switch
                    printstr = (Back.LIGHTBLACK_EX + " " + b[i][j].string + " ")

                if (canMove):
                    printstr = (Back.MAGENTA + " " + b[i][j].string + " ")
            else:
                if (switch):
                    switch = not switch
                    printstr = (Back.LIGHTWHITE_EX + " " + u'\u2003' + " ")

                else:
                    switch = not switch
                    printstr = (Back.LIGHTBLACK_EX + " " + u'\u2003' + " ")

                if (canMove):
                    printstr = (Back.MAGENTA + " " + u'\u2003' + " ")

            print(printstr, end="")
            canMove = False

            sys.stdout.write(Style.RESET_ALL)
        print(i + 1)
        sideNum -= 1
        switch = not switch
    print(bottom)


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
                b[i][j].setX(j)
                b[i][j].setY(i)
    return b


'''
def mover():
    whiteTurn = True
    board = newGame()
    player1 = "White"
    player2 = "Black"
    if (whiteTurn):
        whiteTurn = not whiteTurn
        for i in range(8):
            for j in range(8):
                if board[i][j].toStr(0 == "White"):
'''


def testGame():
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
    b = [[rb0, " ", " ", " ", kb0, " ", " ", rb1],
         [pb0, pb1, " ", pb3, pb4, qw0, " ", pb7],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [pw0, pw1, pb2, qb0, pw4, pw5, pb6, pw7],
         [rw0, " ", " ", " ", kw0, " ", " ", rw1]]
    # rows Y
    for i in range(8):
        # cols X
        for j in range(8):
            if (b[i][j] != " "):
                b[i][j].setX(j)
                b[i][j].setY(i)
    return b


def setPositions(b):
    for i in range(8):
        # cols X
        for j in range(8):
            if (b[i][j] != " "):
                b[i][j].setX(j)
                b[i][j].setY(i)


# the castling test cases are not working. when tried in main, everything was as expected.
def testCases():
    # todo castling DONE ✓
    # todo upgrade a pawn DONE ✓
    # todo put an enemy king in check DONE ✓
    # todo attempt to put a friendly king in check DONE ✓
    # todo putting both kings in check or checkmate at the same time DONE ✓
    # todo make a valid move
    # todo make a invalid move
    # todo capture an enemy piece
    # todo attempt to capture a friendly piece
    # todo castling
    # todo upgrade a pawn
    # todo put an enemy king in check#
    # todo attempt to put a friendly king in check
    # todo putting both kings in check or checkmate at the same time
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

    b = [[rb0, " ", " ", " ", kb0, " ", " ", rb1],
         [pb0, pb1, pb2, pb3, pb4, pb5, pb6, pb7],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [pw0, pw1, pw2, pw3, pw4, pw5, pw6, pw7],
         [rw0, " ", " ", " ", kw0, " ", " ", rw1]]

    setPositions(b)
    # todo castling
    # case 1: move is valid, attempt to castle with all of the rooks
    if not kw0.move(b, rw1.getX(), rw1.getY()):
        print("FAILED TEST CASE 1.1 ATTEMPT A VALID CASTLE KING WITH ROOK")
    if not kw0.move(b, rw0.getX(), rw0.getY()):
        print("FAILED TEST CASE 1.2 ATTEMPT A VALID CASTLE KING WITH ROOK")
    if not kb0.move(b, rb0.getX(), rb0.getY()):
        print("FAILED TEST CASE 1.3 ATTEMPT A VALID CASTLE KING WITH ROOK")
    if not kb0.move(b, rb1.getX(), rb1.getY()):
        print("FAILED TEST CASE 1.4 ATTEMPT A VALID CASTLE KING WITH ROOK")
    b = [[rb0, " ", " ", bw0, kb0, bw0, " ", rb1],
         [pb0, pb1, pb2, pb3, pb4, pb5, pb6, pb7],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [pw0, pw1, pw2, pw3, pw4, pw5, pw6, pw7],
         [rw0, " ", " ", bw0, kw0, bw0, " ", rw1]]
    setPositions(b)
    # case 2: move is NOT valid, attempt to castle with all of the rooks
    if kw0.move(b, rw1.getX(), rw1.getY()):
        print("FAILED TEST CASE 2.1 ATTEMPT A NON VALID CASTLE KING WITH ROOK PIECE IN THE WAY!")
    if kw0.move(b, rw0.getX(), rw0.getY()):
        print("FAILED TEST CASE 2.2 ATTEMPT A NON VALID CASTLE KING WITH ROOK PIECE IN THE WAY!")
    if kb0.move(b, rb0.getX(), rb0.getY()):
        print("FAILED TEST CASE 2.3 ATTEMPT A NON VALID CASTLE KING WITH ROOK PIECE IN THE WAY!")
    if kb0.move(b, rb1.getX(), rb1.getY()):
        print("FAILED TEST CASE 2.4 ATTEMPT A NON VALID CASTLE KING WITH ROOK PIECE IN THE WAY!")
    # CASE 3: NOT VALID KING IS IN CHECK
    b = [[rb0, " ", " ", " ", kb0, " ", " ", rb1],
         [pb0, pb1, pb2, pb3, pb4, qw0, pb6, pb7],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [pw0, pw1, pw2, qb0, pw4, pw5, rb1, pw7],
         [rw0, " ", " ", " ", kw0, " ", " ", rw1]]
    setPositions(b)
    if kw0.move(b, rw1.getX(), rw1.getY()):
        print("FAILED TEST CASE 3.1 ATTEMPT A NON VALID CASTLE KING WITH ROOK. KING IS IN CHECK ")
    if kw0.move(b, rw0.getX(), rw0.getY()):
        print("FAILED TEST CASE 3.2 ATTEMPT A NON VALID CASTLE KING WITH ROOK. PIECE IN THE WAY!")
    # if kb0.move(b, rb0.getX(), rb0.getY()):
    # print("FAILED TEST CASE 3.3 ATTEMPT A NON VALID CASTLE KING WITH ROOK. PIECE IN THE WAY!")
    if kb0.move(b, rb1.getX(), rb1.getY()):
        print("FAILED TEST CASE 3.4 ATTEMPT A NON VALID CASTLE KING WITH ROOK. PIECE IN THE WAY!")
    # CASE 4: A ENEMY PIECE CAN MOVE TO A SQUARE THAT THE KING IS MOVING THROUGH
    b = [[rb0, " ", " ", " ", kb0, " ", " ", " "],
         [pb0, pb1, bw0, pb3, pb4, " ", qw0, pb7],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [pw0, " ", pw2, qb0, pw4, " ", rb1, pw7],
         [rw0, " ", " ", " ", kw0, " ", " ", rw1]]
    setPositions(b)

    if kw0.move(b, rw1.getX(), rw1.getY()):
        print("FAILED TEST CASE 3.5 ATTEMPT A NON VALID CASTLE A ENEMY PIECE CAN MOVE THERE! ")
    if kw0.move(b, rw0.getX(), rw0.getY()):
        print("FAILED TEST CASE 3.6 ATTEMPT A NON VALID CASTLE A ENEMY PIECE CAN MOVE THERE!")
    if kb0.move(b, rb0.getX(), rb0.getY()):
        print("FAILED TEST CASE 3.7 ATTEMPT A NON VALID CASTLE A ENEMY PIECE CAN MOVE THERE!")
    if kb0.move(b, rb1.getX(), rb1.getY()):
        print("FAILED TEST CASE 3.8 ATTEMPT A NON VALID CASTLE A ENEMY PIECE CAN MOVE THERE!")
    b = [[kb0, " ", " ", qw0, " ", " ", " ", " "],
         [pb0, " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", rw1, " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [pw0, pw1, pw2, pw3, pw4, pw5, pw6, pw7],
         [rw0, knw0, bw0, " ", kw0, bw1, knw1, " "]]
    setPositions(b)
    incheckInfo = inCheck(b, "Black")
    chkMate = checkMate(b, "Black", incheckInfo)

    if (incheckInfo[0] == 1):
        if not (chkMate):
            print("FAILED test case 5.1, the black king is in checkmate! ")
    # TO BE CONTINUED

    # WILL THE GAME ALLOW A PLAYER TO MAKE A MOVE THAT WILL AUTOMATICALLY PUT THEM IN CHECK/CHECKMATE?
    b = [[kb0, rb1, " ", " ", " ", " ", " ", qw0],
         [pb0, " ", pb1, qb0, knb0, bb0, " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", rw1, " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [pw0, pw1, pw2, pw3, pw4, pw5, pw6, pw7],
         [rw0, knw0, bw0, " ", kw0, bw1, knw1, " "]]
    setPositions(b)
    if (rb1.move(b, rb1.getX(), rb1.getY() + 1)):
        print("FAILED test case 6.1, player cannot put self in check/checkmate! ")
    b = [[kb0, " ", pb1, " ", " ", " ", " ", qw0],
         [pb0, rb1, " ", qb0, knb0, bb0, " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", rw1, " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [pw0, pw1, pw2, pw3, pw4, pw5, pw6, pw7],
         [rw0, knw0, bw0, " ", kw0, bw1, knw1, " "]]
    setPositions(b)
    if (pb1.move(b, pb1.getX(), pb1.getY() + 1)):
        print("FAILED test case 6.2, player cannot put self in check/checkmate! ")
    b = [[kb0, " ", " ", qb0, " ", " ", " ", qw0],
         [pb0, rb1, " ", " ", knb0, bb0, " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", rw1, " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [pw0, pw1, pw2, pw3, pw4, pw5, pw6, pw7],
         [rw0, knw0, bw0, " ", kw0, bw1, knw1, " "]]
    setPositions(b)
    if (qb0.move(b, qb0.getX(), qb0.getY() + 1)):
        print("FAILED test case 6.3, player cannot put self in check/checkmate! ")
    b = [[kb0, " ", " ", " ", knb0, " ", " ", qw0],
         [pb0, rb1, " ", " ", " ", bb0, " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", rw1, " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [pw0, pw1, pw2, pw3, pw4, pw5, pw6, pw7],
         [rw0, knw0, bw0, " ", kw0, bw1, knw1, " "]]
    setPositions(b)
    if (kb0.move(b, kb0.getX() - 1, kb0.getY() + 2)):
        print("FAILED test case 6.4, player cannot put self in check/checkmate! ")
    b = [[kb0, " ", " ", " ", " ", bb0, " ", qw0],
         [pb0, rb1, " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", rw1, " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [" ", " ", " ", " ", " ", " ", " ", " "],
         [pw0, pw1, pw2, pw3, pw4, pw5, pw6, pw7],
         [rw0, knw0, bw0, " ", kw0, bw1, knw1, " "]]
    setPositions(b)
    if (bb0.move(b, bb0.getX() - 1, bb0.getY() + 1)):
        print("FAILED test case 6.5, player cannot put self in check/checkmate! ")


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


def pawnPromotion(board, xPos, yPos):
    p = ""

    while (p != "q" or p != "r" or p != "b" or p != "r" or p != "k"):

        print("Select a type of piece to promote your pawn to:")
        print("q for Queen")
        print("r for Rook")
        print("b for Bishop")
        print("k for Knight")
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
    if (board[yPos][xPos].getType() == "p"):
        if (enPassant(board, xPos, yPos, moveX, moveY)):
            return makeEnPassant(board, xPos, yPos, moveX, moveY)

    legalMove = board[yPos][xPos].move(board, moveX, moveY)
    castling = False

    '''
    if not pieceInMoveSpot and horizontalMovement and verticalMovement:
        if (self.getPlayer() == "Black"):
            if (moveY == 5 and abs(moveY - self.getY() < 2)):
                if (board[moveY - 1][moveX] != " "):
                    if (board[moveY - 1][moveX].getMoveCount() == 1 and board[moveY - 1][moveX].getType() == "p"):
                        board[moveY][moveX] = board[yPos][xPos]
                        board[moveY][moveX].moveCounter()
                        board[moveY][moveX].updatePos(moveX, moveY)
                        board[moveY - 1][moveX] = " "
                        return
        else:
            if (moveY == 2 and abs(moveY - self.getY() < 2)):
                if (board[moveY + 1][moveX] != " "):
                    if (board[moveY + 1][moveX].getMoveCount() == 1 and board[moveY + 1][moveX].getType() == "p"):
                        board[moveY][moveX] = board[yPos][xPos]
                        board[moveY][moveX].moveCounter()
                        board[moveY][moveX].updatePos(moveX, moveY)
                        board[moveY - 1][moveX] = " "
                        return
        '''

    if (board[moveY][moveX] != " "):
        if (board[yPos][xPos].getType() == "K"
                and board[moveY][moveX].getType() == "r"
                and board[moveY][moveX].getPlayer() == board[yPos][xPos].getPlayer()
        ):
            castling = True

    try:
        if (legalMove and castling):
            return castleMove(board, xPos, yPos, moveX, moveY)
            # check if the move is legal
        if (legalMove):
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

        else:
            # print("That is not a valid move!")
            return board
    except(AttributeError, IndexError):

        # print("That is not a valid move!")
        return board


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


def inCheck(board, player):
    coordinates = findKingPos(board)
    bKingY = coordinates[0]
    bKingX = coordinates[1]
    wkingY = coordinates[2]
    wkingX = coordinates[3]
    r = [0, -1, -1]
    kingX = 0
    kingY = 0

    temp = copy.deepcopy(board)
    if (player == "White"):
        temp[wkingY][wkingX] = " "
        kingX = wkingX
        kingY = wkingY
    elif (player == "Black"):
        # print("We are seeing if black is in CHECK---------------------------------------------------------------------------------------------")
        temp[bKingY][bKingX] = " "
        kingX = bKingX
        kingY = bKingY
    # print("TEMP")
    # printBoard(temp)
    for i in range(8):
        for j in range(8):
            if (temp[i][j] != " "):
                if (temp[i][j].getType() != "K"):
                    if ((temp[i][j].getType() == "p") and (temp[i][j].getPlayer() != player) and temp[i][
                        j].attackMove(temp, kingX, kingY)):
                        r = [1, i, j]
                        return r

                    if temp[i][j].getPlayer() != player and temp[i][j].move(temp, kingX, kingY) and temp[i][
                        j].getType() != "p":
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
                # basically checking every position if it is a friendly piece
                # then if it is a friendly piece and can move, make the move.
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

    # find if there is a move availble to take the king out of check
    return checkMte


def convertInput(input, int):
    x = input[0]
    x = ord(x)
    x = x - 65
    y = int
    y -= 1
    r = [x, y]
    return r


def canEnemyMove(board, player, x, y):
    temp = copy.deepcopy(board)
    for i in range(8):
        for j in range(8):
            if (board[i][j] != " "):
                if (board[i][j].getPlayer() != player) and board[i][j].move(board, x, y):
                    return True

    return False


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


# gets available moves and selects a random move
def randomAImove(board, player):
    list = getAvailableMoves(board, player)
    index = 0
    return random.choice(list)
    # for item in list:
    # mi = list[index]
    # temp = copy.deepcopy(board)
    # temp = makeMove(temp, mi[0], mi[1], mi[2], mi[3])
    # printBoard(temp)
    # temp = copy.deepcopy(board)
    # index += 1


def getAvailableMoves(board, player):
    # holds the coordinates of the piece and the location to move
    # find a piece belonging to player

    mainList = []
    l = [0, 0, 0, 0]
    list = dllist()
    for i in range(8):
        for j in range(8):

            if board[i][j] != " ":

                if board[i][j].getPlayer() == player:

                    # find all the spots on the board that piece can move
                    for y in range(8):
                        for x in range(8):
                            # if board[i][j] != " ":
                            if board[i][j].move(board, x, y):
                                # get the info of the legal move
                                l = [j, i, x, y]
                                # store the info in the main list
                                mainList.append(l)

    return mainList


def showAvailMoves(list, board):
    while (True):
        MI = list.popright()
        board[MI.moveY()][MI.moveX()] = board[MI.getY()][MI.getX()]
        board[MI.getY()][MI.getX()] = " "
        printBoard(board)
        MI = MI.next


class record(object):
    def __init__(self, player1, player2, board):
        self.player1 = player1
        self.player2 = player2
        self.board = board


def getEnemyKingX(piece):
    if piece.getPlayer() == "Black":
        return
    else:
        return


def twoPlayerGame():
    inCheckMate = False
    whiteTurn = True

    while (not inCheckMate):
        try:
            if (whiteTurn):
                print("White Player Select a Piece")
            else:
                print("Black Player Select a Piece ")
            p = input()
            yVal = p[1]
            yVal = int(yVal)
            SelPce = convertInput(p[0].upper(), yVal)
            if (whiteTurn and board[SelPce[0]][SelPce[1]].getPlayer() == "White"):

                printBoardWithAvailMoves(board, SelPce[0], SelPce[1], showAvailMoves)
                print("Select a move spot")
                p = input()
                yVal = p[1]
                yVal = int(yVal)
                move = convertInput(p[0].upper(), yVal)
                yVal = int(yVal)
                board = makeMove(board, SelPce[0], SelPce[1], move[0], move[1])
                whiteTurn = not whiteTurn
            elif not whiteTurn and board[SelPce[0]][SelPce[1]].getPlayer() == "Black":
                printBoardWithAvailMoves(board, SelPce[0], SelPce[1], showAvailMoves)
                print("Select a move spot")
                p = input()
                yVal = p[1]
                yVal = int(yVal)
                move = convertInput(p[0].upper(), yVal)
                yVal = int(yVal)
                board = makeMove(board, SelPce[0], SelPce[1], move[0], move[1])
            else:
                print("you need to select your own piece")

        except(ValueError, NameError, AttributeError, IndexError):
            print("Use correct input")
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

    printBoard(board)
    whiteTurn = not whiteTurn


def twoAIGame():
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
    while (numGames < 10000):
        board = newGame()
        numGames += 1
        winner = ""
        #clear the moveList
        recordMoves.clear()
        counter=0

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
                    inCheckMate = True
                counter += 1
                # return a  random legal move
                mi = randomAImove(board, "White")
                # make the move
                board = makeMove(board, mi[0], mi[1], mi[2], mi[3])
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
                        break

                    else:
                        # print("Black is in Check")
                        whiteInCheck += 1
                    # printBoard(board)

                # return a random legal move
                mi = randomAImove(board, "Black")
                # make the move
                board = makeMove(board, mi[0], mi[1], mi[2], mi[3])
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


        row = [winner, moveCount, moveList,blackInCheck,whiteInCheck,blackWins,whiteWins,numGames]

        with open('AI_data.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)

        csvFile.close()



class driver():
    l = [0, 0, 0]
    list = dllist()
    # ensure that the test cases pass
    # testCases()
    print()
    print()
    # initilize 2d board with pieces
    board = newGame()

    coordinates = findKingPos(board)

    print("Is this game an AI game for data collection? ")
    print("y for yes. any other key for no.")
    ai = input()
    if (ai == "y"):
        # realPlayer = False
        AIPlayer = True
        print("are there 1 or 2 AI Players?")
        numAI = input()
        if (numAI == 1):
            print("is the AI player white or black? w for white or anything else for b")
            color = input()
            if (color == "w"):
                AIColor = "White"
            else:
                AIColor = "Black"


    else:
        AIPlayer = False

    print("Would you like to display available moves when a piece is selected? ")
    print("y for yes. any other key for no.")
    availM = input()
    if (availM == "y"):
        showAvailMoves = True
    else:
        showAvailMoves = False

    # print out the board with pieces and color
    printBoard(board)
    whiteTurn = True
    inChk = False
    inCheckMate = False
    if (AIPlayer):
        inCheckMate = twoAIGame()

    while (not inCheckMate):

        # inCheckMate = twoPlayerGame()

        if (AIPlayer and numAI == 2):
            mi = randomAImove(board, "White")
            board = makeMove(board, mi[0], mi[1], mi[2], mi[3])
            printBoard(board)
            mi = randomAImove(board, "Black")
            board = makeMove(board, mi[0], mi[1], mi[2], mi[3])
            printBoard(board)
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
            # p =
            # if (AIPlayer):

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
