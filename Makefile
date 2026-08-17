.PHONY: help daily dashboard update pull db fill classify lga-leadership signal inspect

help:
	@echo "Targets:"
	@echo "  daily          Refresh local DB with today's data, then print the signal"
	@echo "  dashboard      Refresh local DB with today's data, then open the inspect workbench"
	@echo "  update         Refresh local DB only (pull + db + fill + classify + lga-leadership)"
	@echo "  pull           git pull (fetch today's committed snapshot, if landed)"
	@echo "  db             Load new snapshot/history CSVs into SQLite"
	@echo "  fill           Forward-fill daily price gaps"
	@echo "  classify       Classify stations for today"
	@echo "  lga-leadership Score LGA leadership for today"
	@echo "  signal         Print today's buy/wait signal (no DB refresh)"
	@echo "  inspect        Start the inspect workbench (no DB refresh)"

# Mirrors .github/workflows/daily-db-update.yml: same four steps, same
# no-arg (today) defaults, run locally against fuel_signal.db.
pull:
	git pull

db:
	uv run python -m fuel_signal.db

fill:
	uv run python -m fuel_signal.fill

classify:
	uv run python -m fuel_signal.classify

lga-leadership:
	uv run python -m fuel_signal.lga_leadership

update: pull db fill classify lga-leadership

signal:
	uv run python -m fuel_signal.signal

inspect:
	uv run python -m fuel_signal.inspect

daily: update signal

dashboard: update inspect
