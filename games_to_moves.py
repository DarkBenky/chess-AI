import chess
import chess.pgn

import pandas as pd

def fen_to_array(fen):
    board = chess.Board(fen)
    board_array = [0] * 64
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            flag = 0
            if piece.color == chess.WHITE:
                flag = 1
            board_array[i] = int(str(flag) + str(piece.piece_type))
    return board_array


def games_to_moves():
    X = []
    Y = []

    file = open('games.pgn')
    pgn = chess.pgn.read_game(file)
    all_games = len(list(chess.pgn.read_game(file)))
    while pgn:
        board = pgn.board()
        for move in pgn.mainline_moves():
            X.append(fen_to_array(board.fen()))
            Y.append(move.uci())
            board.push(move)
        pgn = chess.pgn.read_game(file)
        print(f'{len(X)} moves from {len(Y)} games out of {all_games} games')

    df = pd.DataFrame({'X': X, 'Y': Y})
    df.to_csv('games.csv', index=False)

    file.close()
    return X, Y

X, Y = games_to_moves()