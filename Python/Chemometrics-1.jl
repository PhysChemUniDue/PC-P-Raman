function cost(p)
    # Skalierungsfaktoren der Refrenzspektren
    A, B, C, D = p[1:4]
    # Falls das Spektrum eine Hintergrund hat versuchen wir diesen mit zu fitten
    # Dazu nehmen wir an, dass es einen linearen Hintergrund (mx+b) und einen 
    # Gausschen Hintergrund (A*exp(-((x-x0)/sigma)^2)) geben kann
    
    end