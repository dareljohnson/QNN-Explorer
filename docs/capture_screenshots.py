"""Capture README screenshots, driving the real app via Chrome.

Both passes load a saved configuration first (without one, Data/Train/Predict
only render "configure a model first"), and the Train tab additionally needs
data loaded, so each pass loads a demo dataset through the UI.
"""
import pathlib
import sys

from playwright.sync_api import sync_playwright

APP = "http://localhost:8501"
OUT = pathlib.Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)
VIEWPORT = {"width": 1600, "height": 1000}


def select_option(page, label, option, timeout=30_000):
    box = page.locator('div[data-testid="stSelectbox"]').filter(has_text=label).first
    box.locator('div[data-baseweb="select"]').click(timeout=timeout)
    page.wait_for_timeout(700)
    page.get_by_role("option", name=option, exact=True).first.click(timeout=timeout)
    page.wait_for_timeout(2_000)


def choose_radio(page, label, option):
    group = page.locator('div[data-testid="stRadio"]').filter(has_text=label).first
    group.get_by_text(option, exact=True).first.click(timeout=30_000)
    page.wait_for_timeout(2_000)


def click_button(page, name):
    page.get_by_role("button", name=name, exact=False).first.click(timeout=30_000)
    page.wait_for_timeout(2_000)


def open_tab(page, name):
    page.get_by_role("tab", name=name, exact=True).click(timeout=30_000)
    page.wait_for_timeout(2_000)


def load_config(page, filename):
    select_option(page, "Load Existing Configuration", filename)
    click_button(page, "Load Configuration")
    page.wait_for_timeout(3_000)


def shoot(page, filename, needles=(), diagnostic=False):
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(400)
    path = OUT / filename
    page.screenshot(path=str(path))
    text = page.inner_text("body")
    missing = [n for n in needles if n not in text]
    print(f"  {filename:34} {path.stat().st_size // 1024:>4} KB   "
          f"{'OK' if not missing else f'MISSING {missing}'}")
    if missing and diagnostic:
        print("    --- visible text ---")
        print("    " + "\n    ".join(text.splitlines()[:45]))


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=VIEWPORT)
        page.goto(APP, wait_until="load", timeout=60_000)
        page.wait_for_selector('button[role="tab"]', timeout=90_000)
        page.wait_for_timeout(4_000)

        # ------------------------------------------------ CNN pass
        print("CNN pass: config_N4_L2_cnn.json + MNIST demo")
        open_tab(page, "Model Configuration")
        load_config(page, "config_N4_L2_cnn.json")
        shoot(page, "model-configuration.png",
              ["Classical Backbone Type", "Number of Qubits", "Observation Type"])

        open_tab(page, "Data")
        choose_radio(page, "Select Data Source", "Demo")
        select_option(page, "Select Data Type", "Images")
        select_option(page, "Select Demo Images Dataset", "MNIST Digits (Image Classification)")
        click_button(page, "Load and Preprocess Data")
        print("  (waiting for MNIST demo to load)")
        for _ in range(30):
            if "preview" in page.inner_text("body").lower() or "Data Preview" in page.inner_text("body"):
                break
            page.wait_for_timeout(2_000)
        page.wait_for_timeout(6_000)
        shoot(page, "data-preparation.png", ["Select Data Source", "Select Data Type"])

        open_tab(page, "Train")
        page.wait_for_timeout(3_000)
        shoot(page, "training.png", ["Learning Rate", "Batch Size", "Epochs"], diagnostic=True)

        open_tab(page, "Predict")
        page.wait_for_timeout(2_000)
        shoot(page, "predictions.png", ["Instantiate Model"], diagnostic=True)

        # ------------------------------------------------ Transformer pass
        print("Transformer pass: config_N4_L6_transformer.json + ArXiv demo")
        open_tab(page, "Model Configuration")
        load_config(page, "config_N4_L6_transformer.json")
        open_tab(page, "Model Configuration")
        page.wait_for_timeout(2_000)
        shoot(page, "transformer-model-configuration.png",
              ["Classical Backbone Type", "Transformer"])

        open_tab(page, "Data")
        choose_radio(page, "Select Data Source", "Demo")
        select_option(page, "Select Data Type", "CSV")
        select_option(page, "Select Demo CSV Dataset", "ArXiv Scientific Papers (CSV Classification)")
        click_button(page, "Load and Preprocess Data")
        page.wait_for_timeout(10_000)
        shoot(page, "transformer-data-preparation.png",
              ["Direct Text Extraction", "Select Text Column"], diagnostic=True)

        browser.close()


if __name__ == "__main__":
    sys.exit(main())
