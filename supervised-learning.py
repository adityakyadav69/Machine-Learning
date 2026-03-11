from sklearn.linear_model import LinearRegression

X = [[25], [30], [35], [40]]
y = [100, 150, 200, 250]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[32]])

print(prediction)
