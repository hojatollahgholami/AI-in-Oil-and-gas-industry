این مثال براساس دیتاست رویدادهای نادر و نامطلوب چاه های نفت و گاز \یاده سازی شده است. جهت دسترسی به دیتاست از آدرس زیر استفاده شود. فایل ها \س از دانلود. استخراج و در \وشه کاری قرار گیرد.


https://github.com/ricardovvargas/3w_dataset/tree/master/data


class{events_names = {0: 'Normal',
                1: 'Abrupt Increase of BSW',
                2: 'Spurious Closure of DHSV',
                3: 'Severe Slugging',
                4: 'Flow Instability',
                5: 'Rapid Productivity Loss',
                6: 'Quick Restriction in PCK',
                7: 'Scaling in PCK',
                8: 'Hydrate in Production Line'
               }}
               
columns = ['P-PDG':Permanent Downhole Gauge (PDG)

,'P-TPT': Pressure Transducer (TPT)

'T-TPT': Temperature Transducer (TPT)

'P-MON-CKP':

'T-JUS-CKP':

'P-JUS-CKGL':

'T-JUS-CKGL':

'QGL':

'class':events_names]

 The PDG remains fixed in a certain position of the production tubing, and the TPT is part of the subsea Christmas tree. A Downhole Safety Valve (DHSV) and a Production Choke (PCK) are valves and are better explained in the next subsection.
 •
Pressure at the PDG;
•
Pressure at the TPT;
•
Temperature at the TPT;
•
Pressure upstream of the PCK;
•
Temperature downstream of the PCK.
