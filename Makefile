.PHONY: demo clean

demo:
	uv run demo.py --theorem all

clean:
	rm -f figures/*.png
