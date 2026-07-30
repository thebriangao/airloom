import type { Metadata } from "next";
import { AirloomStudio } from "./airloom/AirloomStudio";

export const metadata: Metadata = {
  title: "Airloom | Draw in the space between",
  description:
    "A gesture-driven 3D painting studio. Your hand is the brush and empty space is the canvas.",
};

export default function Home() {
  return <AirloomStudio />;
}
