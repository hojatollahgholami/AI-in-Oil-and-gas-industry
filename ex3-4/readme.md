این مثال براساس داده های تولید محصولات هیدروکربنی در کشورهای مختلف در سال های مختلف \یاده سازی شده است.

ref: https://www.jodidata.org/oil/database/data-downloads.aspx

https://www.jodidata.org/_resources/files/downloads/oil-data/world_primary_csv.zip?iid=163
https://www.jodidata.org/_resources/files/downloads/oil-data/world_secondary_csv.zip?iid=163

Products
(ENERGY_PRODUCT)

World primary table 

Crude oil: CRUDEOIL 

NGL: NGL 

Other: OTHERCRUDE 

Total: TOTCRUDE 

-------------------- 
World secondary table

Liquefied petroleum gases: LPG
Naphtha: NAPHTHA

otor and aviation gasoline: GASOLINE

Kerosenes: KEROSENE

 of which: kerosene type jet fuel: JETKERO
 
Gas/diesel oil: GASDIES

Fuel oil: RESFUEL

Other oil products: ONONSPEC

Total oil products: TOTPRODS 

---------------------
Flows
(FLOW_BREAKDOWN)

World primary table 

Production: INDPROD 

From other sources: OSOURCES 

Imports: TOTIMPSB 

Exports: TOTEXPSB 

Products transferred/Backflows: TRANSBAK 

Direct use: DIRECUSE 

Stock change: STOCKCH 

Statistical difference: STATDIFF 

Refinery intake: REFINOBS 

Closing stocks: CLOSTLV 

---------------------
World secondary table

Refinery output: REFGROUT

Receipts: RECEIPTS

Imports: TOTIMPSB

Exports: TOTEXPSB

Products transferred: PTRANSF

Stock change: STOCKCH

Statistical difference: STATDIFF

Demand: TOTDEMO

Closing stocks: CLOSTLV 

Interproduct transfers: IPTRANSF

---------------------
Countries
(REF_AREA)

Based on ISO 3166-1 alpha-2 standard* available at:
http://www.iso.org/iso/home/store/publication_item.htm?pid=PUB500001%3aen

*Except Kosovo, which currently doesn’t have official attribution of a two-letter code

----------------------
Time
(TIME_PERIOD)

“yyyy-mm” format based on ISO 8601 standard

----------------------
Units
(UNIT_MEASURE)

KBD Thousand Barrels per day (kb/d)

KBBL Thousand Barrels (kbbl)

KL Thousand Kilolitres (kl)

KTONS Thousand Metric Tons (kmt)

CONVBBL Conversion factor barrels/ktons

-----------------------
Colour Codes
(ASSESSMENT_CODE)

1 Results of the assessment show reasonable levels of comparability

2 Consult metadata/Use with caution

3 Data has not been assessed

4 Data under verification 

-----------------------
