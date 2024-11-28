# Example usage
# pgn_string = """
# [Event "Example Game"]
# [Site "Chess.com"]
# [Date "2023.11.27"]
# [Round "1"]
# [White "Player1"]
# [Black "Player2"]
# [Result "1-0"]

# 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0
# """

# pgn_string = """
# [Event "Example Game"]
# [Site "Chess.com"]
# [Date "2023.11.27"]
# [Round "1"]
# [White "Player1"]
# [Black "Player2"]
# [Result "1-0"]

# 1. 1. e4 e5 2. Nf3 Nc6 3. Bb5 Nd4 4. Nxd4 exd4 5. d3 c6 6. Bc4 d6 7. O-O d5 8. exd5
# cxd5 9. Bb3 Nf6 10. Bg5 Be7 11. Re1 O-O 12. Nd2 Bb4 13. a3 Be7 14. Nf3 h6 15.
# Bh4 Re8 16. Nxd4 Bg4 17. f3 Bd7 18. c4 Bc5 19. Rxe8+ Bxe8 20. Bf2 Qb6 21. a4
# Bxd4 22. Bxd4 Qxd4+ 23. Kh1 Qxb2 24. cxd5 Qd4 25. Rc1 Nxd5 26. Rc4 Qb6 27. Rg4
# Ne3 0-1
# """

import chess
import chess.pgn
import io

def generate_fen_from_pgn_string(pgn_string):
    pgn_io = io.StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)
    board = game.board()
    fen_list = [board.fen().split()[0]]
    for move in game.mainline_moves():
        board.push(move)
        fen_list.append(board.fen().split()[0])
    return fen_list

pgn_string = """
[Event "Example Game"]
[Site "Chess.com"]
[Date "2023.11.27"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0
"""

fen_positions = generate_fen_from_pgn_string(pgn_string)

for fen in fen_positions:
    print(fen)

