# # from inference.colors import refree, test_blue, yellow, green, test_white
from inference.colors import red,white,black,blue
# France_filter = {
#     "name": "France",
#     "colors": [test_blue],
# }
#
# Argentina_filter = {
#     "name": "Argentina",
#     "colors": [test_white],
# }
#
# referee_filter = {
#     "name": "Referee",
#     "colors": [refree],
# }
#
# Argentina_goalie_filter = {
#     "name": "Argentina_goalie",
#     "colors": [green],
# }
#
# France_goalie_filter = {
#     "name": "France_goalie",
#     "colors": [yellow],
# }
#
# filters = [
#     France_filter,
#     France_goalie_filter,
#     Argentina_filter,
#     Argentina_goalie_filter,
#     referee_filter,
# ]

# todo 添加球衣颜色
team_red_filter = {
    "name": "team_red",
    "colors": [red],
}

team_white_filter = {
    "name": "team_white",
    "colors": [white],
}

team_black_filter = {
    "name": "team_black",
    "colors": [black],
}
team_blue_filter = {
    "name": "team_blue",
    "colors": [blue],
}
filters = [
    team_red_filter,
    team_white_filter,
    team_black_filter,
    team_blue_filter,
]