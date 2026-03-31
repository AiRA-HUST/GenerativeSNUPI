def save_dna_info_to_bild_file(bild_filename, scaf_coords, scaf_route, stap_coords, stap_routes, unbound_coords):
    """
    Converts dnaInfo into a BILD file format.
    """

    fid = open(bild_filename, 'w')
    num_bps = scaf_coords.size(0)
    scaf_coords = 10 * scaf_coords
    stap_coords = 10 * stap_coords
    unbound_coords = 10 * unbound_coords
    
    # nodes plot
    for i in range(num_bps):
        fid.write('.color steel blue\n')
        fid.write(".sphere\t %.3f\t %.3f\t %.3f\t 1.200 \n" %(scaf_coords[i,0], scaf_coords[i,1], scaf_coords[i,2]))
        
        fid.write('.color orange\n')
        fid.write(".sphere\t %.3f\t %.3f\t %.3f\t 1.200 \n" %(stap_coords[i,0], stap_coords[i,1], stap_coords[i,2]))
        
        fid.write('.color tan\n')
        fid.write(".cylinder\t %.3f\t %.3f\t %.3f\t %.3f\t %.3f\t %.3f\t 0.300 \n" %(scaf_coords[i,0], scaf_coords[i,1], scaf_coords[i,2],
                                                                                     stap_coords[i,0], stap_coords[i,1], stap_coords[i,2]))
        
    
    # edges plot
    for i in range(num_bps-1):
        fid.write('.color steel blue\n')
        src, dst = scaf_route[i], scaf_route[i+1]
        fid.write(".cylinder\t %.3f\t %.3f\t %.3f\t %.3f\t %.3f\t %.3f\t 0.500 \n" %(scaf_coords[src,0], scaf_coords[src,1], scaf_coords[src,2],
                                                                                     scaf_coords[dst,0], scaf_coords[dst,1], scaf_coords[dst,2]))
    for i in range(len(stap_routes)):
        for j in range(stap_routes[i].size(0) - 1):
            fid.write('.color orange\n')
            src, dst = stap_routes[i][j], stap_routes[i][j+1]
            fid.write(".cylinder\t %.3f\t %.3f\t %.3f\t %.3f\t %.3f\t %.3f\t 0.500 \n" %(stap_coords[src,0], stap_coords[src,1], stap_coords[src,2],
                                                                                         stap_coords[dst,0], stap_coords[dst,1], stap_coords[dst,2]))
    
    # unbound nt plot
    for i in range(unbound_coords.size(0)):
        fid.write('.color sea green\n')
        fid.write(".sphere\t %.3f\t %.3f\t %.3f\t 1.200 \n" %(unbound_coords[i,0], unbound_coords[i,1], unbound_coords[i,2]))
    
    fid.close()