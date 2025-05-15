# 1. Importar bibliotecas
from sklearn.datasets import load_diabetes 
import pandas as pd, matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 

# 2. Carregar o dataset como DataFrame 
diab = load_diabetes(as_frame=True) 
df = diab.frame # inclui 10 features + target 
X = df.drop(columns="target") # só as features numéricas

# 3. Instanciar o StandardScaler 
sc = StandardScaler() 

# 4. Ajustar e transformar 
X_std = pd.DataFrame(sc.fit_transform(X), columns=X.columns) 

# 5. Comparar estatísticas 
print("Antes (média / desvio-padrão):\n", X.describe().T[["mean", "std"]]) 
print("\nDepois (deve ficar ≈0 e ≈1):\n", X_std.describe().T[["mean", "std"]]) 

# 6. Visualizar a feature 'bmi' original vs. escalada 
plt.figure() 
X["bmi"].plot(kind="hist", alpha=.5, label="original") 
X_std["bmi"].plot(kind="hist", alpha=.5, label="scaled") 
plt.title("Distribuição do índice de massa corporal (BMI)") 
plt.legend(); plt.show()