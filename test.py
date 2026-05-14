import json
import random
import math
from datetime import datetime, timedelta


class RealisticMarketGenerator:
    def __init__(
        self,
        start_date="2022-01-01",
        days=1500,
        seed=42,
        round_to=50
    ):

        random.seed(seed)

        self.start_date = datetime.strptime(
            start_date,
            "%Y-%m-%d"
        )

        self.days = days
        self.round_to = round_to

        # Starting market prices
        self.base_prices = {
            "Rice": 1000,
            "Tomatoes": 850,
            "Onions": 500,
            "Yam": 750,
            "Beans": 950,
            "Maize": 420,
            "Okra": 500,
            "Etihi": 500,
        }

        # Upper realistic ranges
        self.max_prices = {
            "Rice": 1800,
            "Tomatoes": 1200,
            "Onions": 900,
            "Yam": 2500,
            "Beans": 1950,
            "Maize": 800,
            "Okra": 800,
            "Etihi": 600,
        }

    def round_price(self, price):

        rounded = (
            math.ceil(price / self.round_to)
            * self.round_to
        )

        return max(rounded, 100)

    def generate_prices(self):

        data = {}

        # market-wide economy effect
        market_momentum = 0

        for commodity, start_price in self.base_prices.items():

            prices = []

            current = float(start_price)

            max_price = self.max_prices[commodity]

            total_growth = (
                max_price / start_price
            )

            # commodity-specific trend
            commodity_momentum = 0

            for i in range(self.days):

                date = (
                    self.start_date
                    + timedelta(days=i)
                )

                month = date.month

                progress = i / self.days

                # long-term growth target
                expected = (
                    start_price *
                    (
                        1 +
                        (
                            total_growth - 1
                        )
                        * progress
                    )
                )

                # random market movement
                shock = random.gauss(
                    0,
                    0.01
                )

                # trends persist
                commodity_momentum = (
                    0.85 *
                    commodity_momentum
                    + shock
                )

                # economy inflation effect
                market_momentum = (
                    0.95 *
                    market_momentum
                    + random.gauss(
                        0,
                        0.003
                    )
                )

                # monthly cyclical movement
                seasonal = (
                    0.06 *
                    math.sin(
                        2 *
                        math.pi *
                        i /
                        30
                    )
                )

                # rainy season
                rainy = 0

                if month in [6, 7, 8]:
                    rainy = 0.05

                # harvest season drops
                harvest = 0

                if month in [9, 10]:
                    harvest = -0.08

                # festive period
                festive = 0

                if month == 12:
                    festive = 0.08

                # rare economic shocks
                event = 0

                if random.random() < 0.02:

                    event = random.uniform(
                        -0.15,
                        0.25
                    )

                current = expected * (

                    1
                    + commodity_momentum
                    + market_momentum
                    + seasonal
                    + rainy
                    + harvest
                    + festive
                    + event
                )

                current = max(
                    current,
                    start_price * 0.85
                )

                current = min(
                    current,
                    max_price * 1.08
                )

                final_price = self.round_price(
                    current
                )

                prices.append(
                    {
                        "date":
                        date.strftime(
                            "%Y-%m-%d"
                        ),

                        "price":
                        final_price
                    }
                )

            data[commodity] = prices

        return data

    def save_json(
        self,
        filename="price_data.json"
    ):

        data = self.generate_prices()

        with open(
            filename,
            "w"
        ) as f:

            json.dump(
                data,
                f,
                indent=2
            )

        print(
            f"\n✅ Generated {filename}"
        )

        print(
            f"Days: {self.days}"
        )

        print(
            "Includes:"
        )

        print(
            "- seasonal patterns"
        )

        print(
            "- rainy season effects"
        )

        print(
            "- festive spikes"
        )

        print(
            "- market shocks"
        )

        print(
            "- trend persistence"
        )

        print(
            "- inflation movement"
        )

        return data


# RUN

if __name__ == "__main__":

    generator = (
        RealisticMarketGenerator(

            start_date="2022-01-01",

            days=1500,

            seed=42,

            round_to=50
        )
    )

    price_data = (
        generator.save_json(
            "price_data.json"
        )
    )

    print(
        "\nRice sample:"
    )

    for p in price_data[
        "Rice"
    ][:5]:

        print(p)

    print("...")

    for p in price_data[
        "Rice"
    ][-5:]:

        print(p)