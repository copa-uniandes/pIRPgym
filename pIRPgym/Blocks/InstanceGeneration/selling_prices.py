from numpy.random import seed, random, randint, lognormal

class selling_prices():

    @staticmethod
    def get_selling_prices(inst_gen,discount) -> dict:
        #discount = kwargs["discount"]
        sell_prices = dict()

        if  discount[0] == "no": sell_prices = selling_prices.gen_sell_price_null_discount(inst_gen)
        elif discount[0] == "lin": sell_prices = selling_prices.gen_sell_price_linear_discount(inst_gen)
        elif discount[0] == "mild": sell_prices = selling_prices.gen_sell_price_mild_discount(inst_gen, discount[1])
        elif discount[0] == "strong": sell_prices = selling_prices.gen_sell_price_strong_discount(inst_gen, discount[1])

        return sell_prices
    
    @staticmethod
    def gen_salvage_price(inst_gen, **kwargs) -> dict:
        salv_price = dict()
        for k in inst_gen.Products:
            li = []
            for t in inst_gen.Horizon:
                for i in inst_gen.Suppliers:
                    if i in inst_gen.M_kt[k,t]:
                        li += [inst_gen.W_p[t][i,k]]
            salv_price[k] = sum(li)/len(li)
        
        return salv_price
    
    @staticmethod
    def gen_optimal_price(inst_gen, **kwargs) -> dict:
        opt_price = {}
        for k in inst_gen.Products:            
            opt_price[k] = 20*inst_gen.salv_price[k]
        
        return opt_price

    @staticmethod
    def gen_sell_price_strong_discount(inst_gen, conv_discount) -> dict:

        def ff(k):
            return k*(k+1)/2

        sell_prices = dict()
        for k in inst_gen.Products:
            for o in range(inst_gen.O_k[k] + 1):
                if conv_discount == "conc":
                    if o == inst_gen.O_k[k]: sell_prices[k,o] = inst_gen.salv_price[k]
                    else: 
                        sell_prices[k,o] = inst_gen.opt_price[k] - ((inst_gen.opt_price[k]-inst_gen.salv_price[k])*0.25)*(ff(o+1)-1)/(ff(inst_gen.O_k[k])-1)
                elif conv_discount == "conv":
                    if o == 0: sell_prices[k,o] = inst_gen.opt_price[k]
                    else: sell_prices[k,o] = inst_gen.salv_price[k] + ((inst_gen.opt_price[k]-inst_gen.salv_price[k])*0.25)*(ff(inst_gen.O_k[k]-o+1)-1)/(ff(inst_gen.O_k[k])-1)
        
        return sell_prices


    @staticmethod   
    def gen_sell_price_mild_discount(inst_gen, conv_discount) -> dict:

        def ff(k):
            return k*(k+1)/2
        
        sell_prices = dict()
        for k in inst_gen.Products:
            for o in range(inst_gen.O_k[k] + 1):
                if conv_discount == "conc":
                    sell_prices[k,o] = inst_gen.salv_price[k] + (inst_gen.opt_price[k]-inst_gen.salv_price[k])*(ff(inst_gen.O_k[k])-ff(o))/ff(inst_gen.O_k[k])
                elif conv_discount == "conv":
                    sell_prices[k,o] = inst_gen.salv_price[k] + (inst_gen.opt_price[k]-inst_gen.salv_price[k])*(ff(inst_gen.O_k[k]-o))/ff(inst_gen.O_k[k])
        
        return sell_prices


    @staticmethod
    def gen_sell_price_null_discount(inst_gen, **kwargs) -> dict:

        sell_prices = {(k,o):inst_gen.opt_price[k] for k in inst_gen.Products for o in range(inst_gen.O_k[k] + 1)}
        return sell_prices
    

    @staticmethod
    def gen_sell_price_linear_discount(inst_gen, **kwargs) -> dict:
        
        sell_prices = dict()
        for k in inst_gen.Products:
            for o in range(inst_gen.O_k[k] + 1):
                sell_prices[k,o] = inst_gen.salv_price[k] + (inst_gen.opt_price[k]-inst_gen.salv_price[k])*(inst_gen.O_k[k]-o)/inst_gen.O_k[k]
        
        return sell_prices

      