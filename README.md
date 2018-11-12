# SC2-Q-Learning
Codes on SCLE 2.0 with Zerg race using Q-learning under development

Código en SCLE 2.0 con raza Zerg usando Q-learning en desarrollo

Hasta ahora se cuenta con una simple guía de intalación para realizar las instalaciones pertinentes, archivo "Guía de instalación.pdf".
  Se mencionan las fuentes y citios de referencia consultadas para mayor referencia y consulta en caso de dudas o problemas en el proceso   de instalación.

Igualmente, el codigo actual se encuentra comentado en español, se comparten los archivos donde se guarda la tabla-Q; los archivos .gz y .csv corresponden al valor de la variable 'DATA_FILE' dentro de cada codigo .py (las tablas Q de un codigo no son compatibles con otro).

En caso de querer referencias adicionales que puedan ayudar a desarrollar otro bot por tu cuenta recomiendo los tutoriales y repositorios de "Steven Brown", los cuales puedes encontar en:

https://github.com/skjb/pysc2-tutorial

Aprovecho para hacer un agradecimiento especial a "Steven Brown" por sus tutoriales y explicaciones en los cuales me he basado para escribir el Bot aquí presentado. Thank you very much "Steven Brown" for your tutorials and codes!.

Nota: El Bot '20181031_Q-table_SC2_1.py' no logra ganar muchos juegos contra el Bot de SC2 en la dificultad very-easy, sin embargo el bot '20181111_SC2_Q-table_RSR_obs-act.py' gana la mayoria, en este codigo se agrego a las observaciones una representación de 4x4 del minimapa (en un arreglo) indicando donde se ven unidades enemigas y otro donde se indican nuestras unidades; aparentemente lo que casusa la mayor parte de la mejoria es limitar las acciones de ataque a donde se ven unidades enemigas y solo cuando el Bot no ve unidades enemigas puede explorar el mapa.

Saludos a todos!
