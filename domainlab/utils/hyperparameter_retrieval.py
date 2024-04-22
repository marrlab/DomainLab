"""
retrieval for hyperparameters
"""

def get_gamma_reg(args, model_name):
    """
    Retrieves either a shared gamma regularization, or individual ones for each specified object
    """
    gamma_reg = args.gamma_reg
    print(gamma_reg)
    if isinstance(gamma_reg, dict):
        print("is instance dict")
        if model_name in gamma_reg: 
            return gamma_reg[model_name]  
        elif 'default' in gamma_reg: 
            return gamma_reg['default']
        else: 
            raise ValueError("If a gamma_reg dict is specified, but no value set for every model and trainer, a default value must be specified.")
    else: 
        return gamma_reg  # Return the single value if it's not a dictionary
