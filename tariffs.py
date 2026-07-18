from datetime import datetime

def electricity_price(hour, weekday, cycle_type="Two-cycle", price_model="Fixed"):
    """
    Calculates the electricity price based on hour and tariff cycle.
    In Portugal, there are different tariffs depending on the cycle type (Simple, Two-cycle, or Three-cycle) and the price model (Fixed or Variable).
    """

    cycle_alias = {
        "Simples": "Simple",
        "Bi-horária": "Two-cycle",
        "Biciclo": "Two-cycle",
        "Tri-horária": "Three-cycle",
        "Triciclo": "Three-cycle",
    }
    price_alias = {
        "Preço fixo": "Fixed",
        "Fixo": "Fixed",
        "Preço variável": "Variable",
        "Variável": "Variable",
    }

    cycle_type = cycle_alias.get(cycle_type, cycle_type)
    price_model = price_alias.get(price_model, price_model)

    if price_model == "Fixed":
        base_price = 0.18 # fictitious value

    else:
        # Aqui no futuro podes ligar ao OMIE, para já simulamos um indexado
        # Simulated indexed price
        base_price = 0.14 if (hour < 7 or hour > 23) else 0.20

    if cycle_type == "Simple":
        return base_price
    
    elif cycle_type == "Two-cycle":
        # Off-peak: 22h to 08h and weekends
        if (hour >= 22 or hour < 8) or weekday >= 5:
            return base_price * 0.5 #50% cheaper nothing fixed

        else:
            return base_price * 1.2 #20% more expensive nothing fixed

    elif cycle_type == "Three-cycle":
        # Peak, more expensive
        if (9 <= hour <12) or (18 <= hour < 21):
            return base_price * 2.0

        # Off-peak, cheaper
        elif (0 <= hour < 7):
            return base_price * 0.4
        
        # Full, normal price
        else:
            return base_price #normal