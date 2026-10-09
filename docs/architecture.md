\# Architecture



```text

CCTV Video

&#x20;   |

&#x20;   v

YOLOv8 Detection

&#x20;   |

&#x20;   v

Person Tracking

&#x20;   |

&#x20;   v

Event Generator

&#x20;   |

&#x20;   v

SQLite Database

&#x20;   |

&#x20;   +-------------------+

&#x20;   |                   |

&#x20;   v                   v

Analytics Engine    FastAPI APIs

&#x20;   |                   |

&#x20;   +---------+---------+

&#x20;             |

&#x20;             v

&#x20;     Streamlit Dashboard

```



\## Components



1\. YOLOv8

&#x20;  - Detects people from CCTV footage.



2\. Person Tracking

&#x20;  - Assigns IDs to visitors.



3\. Event Generator

&#x20;  - Generates ENTRY, EXIT, REENTRY events.



4\. SQLite Database

&#x20;  - Stores all events.



5\. Analytics Engine

&#x20;  - Calculates KPIs, crowding, funnel metrics, dwell time.



6\. FastAPI

&#x20;  - Exposes production-ready APIs.



7\. Streamlit Dashboard

&#x20;  - Displays live store intelligence metrics.

