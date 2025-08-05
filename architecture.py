

V1 = Layer()
V2 = Layer()
V3 = Layer()

V1.update()
V2.update()
net.connection(V1, V2, TwoWayAreaConnection)
net.connection(V2, V3, TwoWayAreaConnection)




# decisions
GPeOut -= matGo
GPeIn -= matNoGo
GPeIn -= GPeOut
GPeTA -= GPeIn
inhibitMat = avg(GPeTA)
matGo -= inhibitMat
matNoGo -= inhibitMat
GPi -= matGo
GPi -= GPeIn
net.connection(matNoGo, GPeIn, InhibitOneToOneConnection)
net.connection(matNoGo, GPeIn, InhibitOneToOneConnection)


