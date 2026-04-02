import pandas as pd


def generate_data() -> pd.DataFrame:
	data = {
		"t": [0, 0, 0, 1, 1, 1],
		"x": [0, 0, 1, 0, 0, 1],
		"y": [200, 120, 300, 500, 600, 800],
	}
	return pd.DataFrame(data)


if __name__ == "__main__":
	print(generate_data())
